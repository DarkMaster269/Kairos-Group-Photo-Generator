"""
detector.py — Rotation-augmented, MediaPipe-only face detection and landmarking.

History: originally used DeepFace/MTCNN at 3 rotation angles per image, which
pulled in TensorFlow and was heavy enough to hang or get OOM-killed on a small
hosting container. That was replaced with a single-pass MediaPipe BlazeFace
detector, which fixed the hang but — because BlazeFace alone only reliably
finds roughly-upright, unoccluded faces — dropped detection count badly on
crowded, tilted-head group photos.

This version keeps the light MediaPipe-only stack (no TensorFlow) but restores
rotation augmentation: BlazeFace is cheap enough that running it 3x per image
is still far lighter than a single MTCNN/TensorFlow pass was. Detections across
angles are deduplicated with IoU-based NMS, then MediaPipe FaceLandmarker runs
per detected crop for the 478-point mesh, same as before.

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

LANDMARKER_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# full_range (not short_range) — short_range is tuned for close selfie-camera
# faces within ~2m; full_range handles the smaller, further-away faces you get
# in a typical group photo.
DETECTOR_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)

CROP_PADDING_PCT: float = 0.35
"""Padding added to each side of the face crop to capture jawline and hair details."""

MIN_DETECTION_CONFIDENCE: float = 0.25
"""Minimum confidence for the BlazeFace detector. Lowered from an earlier 0.5 —
BlazeFace's confidence scale runs lower than MTCNN's did for partially-occluded
or off-angle faces in a crowded group photo, so 0.5 was silently dropping valid
faces. 0.25 roughly matches the old detector's effective sensitivity."""

ROTATION_ANGLES: Tuple[float, ...] = (0, -20, 20)
"""Rotation angles (degrees) to run BlazeFace at, to catch tilted/leaning heads
in a group photo. BlazeFace is cheap enough that running it 3x per image is
still far lighter than a single MTCNN/TensorFlow pass was."""

IOU_THRESHOLD: float = 0.3
"""Overlap threshold for Non-Maximum Suppression (NMS) deduplication across
rotation passes."""


# ── Rotation helpers ─────────────────────────────────────────────────────────

def _rotate_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate image by angle degrees and return the rotated image and rotation matrix."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return np.ascontiguousarray(rotated), M


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


def _pad_to_square(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pad an image with black borders to make it square, preserving aspect ratio."""
    h, w = image.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return padded, x_offset, y_offset


# ── FaceMeshDetector ─────────────────────────────────────────────────────────

class FaceMeshDetector:
    """
    Wraps MediaPipe's Face Detector (BlazeFace, run at multiple rotation angles)
    and FaceLandmarker (478-point mesh per face) to detect faces and landmarks
    in a group photo.

    Each face dict returned contains:
        - ``bbox``: ``(x_min, y_min, width, height)`` in absolute pixels.
        - ``landmarks``: list of ``(x, y, z)`` tuples mapped to full image space.
    """

    def __init__(
        self,
        max_num_faces: int = 15,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        refine_landmarks: bool = True,
    ) -> None:
        pipeline_dir = os.path.dirname(os.path.abspath(__file__))
        self._landmarker_path = os.path.join(pipeline_dir, "face_landmarker.task")
        self._detector_path = os.path.join(pipeline_dir, "blaze_face_full_range.tflite")

        self._ensure_model_exists(self._landmarker_path, LANDMARKER_MODEL_URL)
        self._ensure_model_exists(self._detector_path, DETECTOR_MODEL_URL)

        detector_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=self._detector_path),
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(detector_options)

        landmarker_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self._landmarker_path),
            num_faces=1,  # each cropped region contains exactly one face
            min_face_detection_confidence=0.1,  # low limit — bbox already found it
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)
        logger.debug("FaceMeshDetector (rotation-augmented, MediaPipe-only) initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all faces in *image* and return per-face landmark data.

        Args:
            image: ``np.ndarray`` of shape ``(H, W, 3)`` in BGR format (OpenCV default).

        Returns:
            A list of face dictionaries, one per detected face.
        """
        if image is None or image.ndim < 2:
            raise ValueError("image must be a non-None numpy array.")

        h_img, w_img = image.shape[:2]
        rgb_full = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Run BlazeFace at multiple rotation angles to find tilted/leaning faces
        all_raw_faces: List[Dict[str, Any]] = []

        for angle in ROTATION_ANGLES:
            if angle == 0:
                rotated = rgb_full
                M = None
            else:
                rotated, M = _rotate_image(rgb_full, angle)

            mp_rot_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rotated)
            detection_result = self._detector.detect(mp_rot_image)

            for detection in detection_result.detections:
                bb = detection.bounding_box
                score = detection.categories[0].score if detection.categories else 1.0

                rx, ry, rw, rh = bb.origin_x, bb.origin_y, bb.width, bb.height
                corners = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]

                if angle != 0:
                    orig_corners = [_rotate_point(pt, M, invert=True) for pt in corners]
                else:
                    orig_corners = corners

                orig_xs = [pt[0] for pt in orig_corners]
                orig_ys = [pt[1] for pt in orig_corners]
                ox_min = max(0, min(orig_xs))
                oy_min = max(0, min(orig_ys))
                ox_max = min(w_img, max(orig_xs))
                oy_max = min(h_img, max(orig_ys))

                bbox = (int(ox_min), int(oy_min), int(ox_max - ox_min), int(oy_max - oy_min))
                if bbox[2] > 5 and bbox[3] > 5:
                    all_raw_faces.append({"bbox": bbox, "confidence": score})

        # 2. Deduplicate across rotation passes with IoU-based NMS
        all_raw_faces.sort(key=lambda f: f["confidence"], reverse=True)
        unique_bboxes: List[Tuple[int, int, int, int]] = []
        for face in all_raw_faces:
            box = face["bbox"]
            if not any(_compute_iou(box, u) > IOU_THRESHOLD for u in unique_bboxes):
                unique_bboxes.append(box)

        # 3. For each distinct bounding box, crop, pad to square, and run FaceLandmarker
        faces: List[Dict[str, Any]] = []
        for bbox in unique_bboxes:
            x, y, w_box, h_box = bbox

            pad_w = int(w_box * CROP_PADDING_PCT)
            pad_h = int(h_box * CROP_PADDING_PCT)

            crop_x1 = max(0, x - pad_w)
            crop_y1 = max(0, y - pad_h)
            crop_x2 = min(w_img, x + w_box + pad_w)
            crop_y2 = min(h_img, y + h_box + pad_h)

            crop = rgb_full[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue

            crop_square, pad_x, pad_y = _pad_to_square(crop)
            square_size = crop_square.shape[0]
            crop_resized = cv2.resize(crop_square, (256, 256), interpolation=cv2.INTER_CUBIC)

            mp_crop_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_resized)
            result = self._landmarker.detect(mp_crop_image)

            if not result.face_landmarks:
                logger.debug("FaceLandmarker found no mesh inside bbox %s", bbox)
                continue

            face_lms = result.face_landmarks[0]

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

        logger.info("Rotation-augmented MediaPipe detector extracted %d face(s).", len(faces))
        return faces

    def close(self) -> None:
        """Release the underlying MediaPipe task resources."""
        self._detector.close()
        self._landmarker.close()

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_exists(self, path: str, url: str) -> None:
        """Downloads a model asset file if it isn't already bundled/cached."""
        if not os.path.exists(path):
            logger.info("Model not found at %s. Downloading from %s...", path, url)
            try:
                urllib.request.urlretrieve(url, path)
                logger.info("Successfully downloaded model to %s", path)
            except Exception as e:
                logger.error("Failed to download model asset from %s", url)
                raise RuntimeError(
                    f"MediaPipe Tasks API requires {os.path.basename(path)} but "
                    f"download failed: {e}"
                ) from e
