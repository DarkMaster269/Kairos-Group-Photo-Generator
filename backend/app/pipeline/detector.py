"""
detector.py — Lightweight MediaPipe-only face detection and landmarking.

Replaces the earlier DeepFace/MTCNN hybrid detector. That version ran MTCNN
(via TensorFlow) at three rotation angles per image to catch tilted faces —
correct in principle, but far too heavy to run reliably on a small hosting
container, which is what was causing the pipeline to hang or get killed right
after Gate 1.

This version uses MediaPipe's own Face Detector (BlazeFace, full-range model —
better suited than short-range for group photos where faces are smaller/further
from camera) to get bounding boxes directly on the full image, then runs the
existing FaceLandmarker per crop for the 478-point mesh. No TensorFlow, no
per-image rotation passes.

Tradeoff: this drops robustness to heavily tilted/rotated faces that the
rotation-augmented MTCNN pass was designed to catch. For a group photo where
people are roughly upright, that's a fine tradeoff for reliably not hanging.
If you want tilt-robustness back later, it's worth revisiting on a bigger
Railway plan rather than the free/hobby tier.

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

MIN_DETECTION_CONFIDENCE: float = 0.5
"""Minimum confidence for the BlazeFace bounding-box detector."""


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    Wraps MediaPipe's Face Detector (bounding boxes) and FaceLandmarker
    (478-point mesh per face) to detect faces and landmarks in a group photo.

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
        logger.debug("FaceMeshDetector (MediaPipe-only) initialized.")

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
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Detect bounding boxes on the full image in one pass
        mp_full_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self._detector.detect(mp_full_image)

        faces: List[Dict[str, Any]] = []

        for detection in detection_result.detections:
            bb = detection.bounding_box
            x, y, w_box, h_box = bb.origin_x, bb.origin_y, bb.width, bb.height

            # Clamp to image bounds defensively
            x = max(0, x)
            y = max(0, y)
            w_box = min(w_box, w_img - x)
            h_box = min(h_box, h_img - y)
            if w_box <= 5 or h_box <= 5:
                continue

            # 2. Pad, crop, square, resize, then run FaceLandmarker on the crop
            pad_w = int(w_box * CROP_PADDING_PCT)
            pad_h = int(h_box * CROP_PADDING_PCT)

            crop_x1 = max(0, x - pad_w)
            crop_y1 = max(0, y - pad_h)
            crop_x2 = min(w_img, x + w_box + pad_w)
            crop_y2 = min(h_img, y + h_box + pad_h)

            crop = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue

            crop_square, pad_x, pad_y = _pad_to_square(crop)
            square_size = crop_square.shape[0]
            crop_resized = cv2.resize(crop_square, (256, 256), interpolation=cv2.INTER_CUBIC)

            mp_crop_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_resized)
            result = self._landmarker.detect(mp_crop_image)

            if not result.face_landmarks:
                logger.debug("FaceLandmarker found no mesh inside bbox %s", (x, y, w_box, h_box))
                continue

            face_lms = result.face_landmarks[0]
            scale = square_size / 256.0

            orig_landmarks: List[Tuple[float, float, float]] = []
            for lm in face_lms:
                x_sq = lm.x * 256 * scale
                y_sq = lm.y * 256 * scale
                x_crop = x_sq - pad_x
                y_crop = y_sq - pad_y
                x_orig = x_crop + crop_x1
                y_orig = y_crop + crop_y1
                z_orig = lm.z * square_size
                orig_landmarks.append((x_orig, y_orig, z_orig))

            faces.append({"bbox": (x, y, w_box, h_box), "landmarks": orig_landmarks})

        logger.info("MediaPipe detector extracted %d face(s).", len(faces))
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
