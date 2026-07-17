"""
detector.py — Hybrid Rotation-Augmented MTCNN + MediaPipe Face Mesh.

Uses MTCNN (via DeepFace) at multiple rotation angles (0, -20, 20 degrees)
to locate all human faces in a group photo (including heavily tilted/leaning
faces). Crops, pads to square (preserving aspect ratio), and runs MediaPipe
FaceLandmarker on each face crop individually to extract 478 landmarks.

Projects landmarks back to the original image coordinate space.

Part of the Kairos CV pipeline: detect → cluster → score → blend → gate.
"""

import logging
import os
import urllib.request
from typing import Any, Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
"""Google's official Face Landmarker model task file URL."""

FACE_DETECTION_CONFIDENCE: float = 0.3
"""Minimum confidence score to accept an MTCNN face detection."""

CROP_PADDING_PCT: float = 0.3
"""Padding added to each side of the face crop to capture jawline and hair details."""

IOU_THRESHOLD: float = 0.3
"""Overlap threshold for Non-Maximum Suppression (NMS) deduplication."""


# ── Rotation Helpers ─────────────────────────────────────────────────────────

def _rotate_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate image by angle degrees and return the rotated image and rotation matrix."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Translate rotation center to center of new boundaries
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated, M


def _rotate_point(pt: Tuple[float, float], M: np.ndarray, invert: bool = False) -> Tuple[float, float]:
    """Map a point through the affine rotation matrix. If invert, reverse the transform."""
    if invert:
        M_inv = cv2.invertAffineTransform(M)
        x = pt[0] * M_inv[0, 0] + pt[1] * M_inv[0, 1] + M_inv[0, 2]
        y = pt[0] * M_inv[1, 0] + pt[1] * M_inv[1, 1] + M_inv[1, 2]
        return (x, y)
    else:
        x = pt[0] * M[0, 0] + pt[1] * M[0, 1] + M[0, 2]
        y = pt[0] * M[1, 0] + pt[1] * M[1, 1] + M[1, 2]
        return (x, y)


def _pad_to_square(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pad an image with black borders to make it square, preserving aspect ratio."""
    h, w = image.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    return padded, x_offset, y_offset


def _compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) of two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    if boxAArea + boxBArea - interArea <= 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)


# ── FaceMeshDetector ─────────────────────────────────────────────────────────

class FaceMeshDetector:
    """
    Wraps MTCNN and MediaPipe FaceLandmarker to detect faces and landmarks.

    Detects faces globally using MTCNN at multiple rotation angles (0, -20, 20
    degrees) to handle tilted faces in group photos. Filters and deduplicates
    using NMS, then runs MediaPipe FaceLandmarker on square face crops.

    Each face dict returned contains:
        - ``bbox``: ``(x_min, y_min, width, height)`` in absolute pixels.
        - ``landmarks``: list of ``(x, y, z)`` tuples mapped to full image space.

    Args:
        max_num_faces:            Ignored (retained for backward compatibility).
        min_detection_confidence: Ignored (retained for backward compatibility).
        refine_landmarks:         Ignored (retained for backward compatibility).
    """

    def __init__(
        self,
        max_num_faces: int = 10,
        min_detection_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        self._model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "face_landmarker.task",
        )
        self._ensure_model_exists()

        # Setup FaceLandmarker options
        base_options = python.BaseOptions(model_asset_path=self._model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,  # Each cropped region contains exactly one face
            min_face_detection_confidence=0.1,  # Low confidence limit on cropped face
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        logger.debug("FaceMeshDetector hybrid rotation-augmented initializer complete.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all faces in *image* and return per-face landmark data.

        Args:
            image: ``np.ndarray`` of shape ``(H, W, 3)`` in BGR format (OpenCV default).

        Returns:
            A list of face dictionaries, one per detected face::

                [
                    {
                        "bbox":      (x_min, y_min, width, height),
                        "landmarks": [(x, y, z), ...],
                    },
                    ...
                ]
        """
        if image is None or image.ndim < 2:
            raise ValueError("image must be a non-None numpy array.")

        h_img, w_img = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lazy import of DeepFace to keep load times fast
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise ImportError(
                "DeepFace is required for the hybrid face detector. "
                "Install it with: pip install deepface tf-keras"
            ) from exc

        # 1. Run MTCNN at multiple rotation angles to find tilted faces
        all_raw_faces = []
        angles = [0, -20, 20]
        
        for angle in angles:
            if angle == 0:
                rotated = image.copy()
                M = np.eye(2, 3, dtype=np.float32)
            else:
                rotated, M = _rotate_image(image, angle)
                
            try:
                results = DeepFace.extract_faces(
                    img_path=rotated,
                    detector_backend='mtcnn',
                    enforce_detection=False
                )
                detected = [r for r in results if r["confidence"] > FACE_DETECTION_CONFIDENCE]
                
                for r in detected:
                    area = r["facial_area"]
                    rx, ry, rw, rh = area["x"], area["y"], area["w"], area["h"]
                    
                    # Get corners
                    corners = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
                    if angle != 0:
                        orig_corners = [_rotate_point(pt, M, invert=True) for pt in corners]
                    else:
                        orig_corners = corners
                        
                    # Extract bounding box in original image space
                    orig_xs = [pt[0] for pt in orig_corners]
                    orig_ys = [pt[1] for pt in orig_corners]
                    ox_min = max(0, min(orig_xs))
                    oy_min = max(0, min(orig_ys))
                    ox_max = min(w_img, max(orig_xs))
                    oy_max = min(h_img, max(orig_ys))
                    
                    bbox = (int(ox_min), int(oy_min), int(ox_max - ox_min), int(oy_max - oy_min))
                    if bbox[2] > 5 and bbox[3] > 5:
                        all_raw_faces.append({
                            "bbox": bbox,
                            "confidence": r["confidence"]
                        })
            except Exception as e:
                logger.error("Face extraction failed for angle %d: %s", angle, e)

        # 2. Apply Non-Maximum Suppression (NMS) to deduplicate bounding boxes
        all_raw_faces.sort(key=lambda x: x["confidence"], reverse=True)
        unique_bboxes = []
        for face in all_raw_faces:
            box = face["bbox"]
            is_dup = False
            for u_box in unique_bboxes:
                if _compute_iou(box, u_box) > IOU_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                unique_bboxes.append(box)

        # 3. For each distinct bounding box, crop, pad to square, and run FaceMesh
        faces: List[Dict[str, Any]] = []
        for idx, bbox in enumerate(unique_bboxes):
            x, y, w_box, h_box = bbox
            
            # Add padding around the box
            pad_w = int(w_box * CROP_PADDING_PCT)
            pad_h = int(h_box * CROP_PADDING_PCT)
            
            crop_x1 = max(0, x - pad_w)
            crop_y1 = max(0, y - pad_h)
            crop_x2 = min(w_img, x + w_box + pad_w)
            crop_y2 = min(h_img, y + h_box + pad_h)
            
            crop = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue
                
            # Pad to square to avoid aspect ratio stretching
            crop_square, pad_x, pad_y = _pad_to_square(crop)
            square_size = crop_square.shape[0]
            
            # Resize to 256x256
            crop_resized = cv2.resize(crop_square, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            # Run MediaPipe FaceMesh
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_resized)
            result = self._landmarker.detect(mp_image)
            
            if result.face_landmarks:
                face_lms = result.face_landmarks[0]
                
                # Project landmarks back to original image space
                orig_landmarks: List[Tuple[float, float, float]] = []
                for lm in face_lms:
                    x_sq = lm.x * square_size
                    y_sq = lm.y * square_size
                    x_crop = x_sq - pad_x
                    y_crop = y_sq - pad_y
                    x_orig = x_crop + crop_x1
                    y_orig = y_crop + crop_y1
                    z_orig = lm.z * square_size
                    orig_landmarks.append((x_orig, y_orig, z_orig))
                
                faces.append({"bbox": bbox, "landmarks": orig_landmarks})
                logger.debug("Successfully extracted mesh for bbox %s", bbox)
            else:
                logger.debug("FaceMesh landmarker failed inside crop for bbox %s", bbox)

        logger.info("Hybrid detector successfully extracted %d face(s).", len(faces))
        return faces

    def close(self) -> None:
        """Release the underlying FaceLandmarker resources."""
        self._landmarker.close()

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_exists(self) -> None:
        """Downloads the face_landmarker.task model file if missing."""
        if not os.path.exists(self._model_path):
            logger.info(
                "Face landmarker model not found at %s. Downloading...",
                self._model_path,
            )
            try:
                urllib.request.urlretrieve(MODEL_URL, self._model_path)
                logger.info("Successfully downloaded model to %s", self._model_path)
            except Exception as e:
                logger.error("Failed to download model asset from %s", MODEL_URL)
                raise RuntimeError(
                    f"MediaPipe Tasks API requires face_landmarker.task but "
                    f"download failed: {e}"
                ) from e
