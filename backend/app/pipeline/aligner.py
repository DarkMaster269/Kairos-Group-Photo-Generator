"""
aligner.py — Face alignment via similarity transformation.

Given a source image and corresponding source/destination MediaPipe face
landmarks, computes a 2D similarity transform (rotation + uniform scale +
translation, no shear) that maps the source face landmark positions to the
destination face landmark positions, then warps the entire source image so
the face aligns geometrically with the base frame.

Uses expression-stable landmark points — outer/inner eye corners, nose
bridge top/tip, and mouth corners — so the transform is driven by the
rigid skull structure and is unaffected by smiling, blinking, or gaze.

Part of the Kairos CV pipeline: detect → cluster → score → **align** → blend → gate.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Expression-stable landmark indices (MediaPipe 478-point model) ────────────
#
# These points sit on the bony skull structure and move very little with
# smiles, blinks, or gaze changes. Using both sides of the face gives the
# solver enough spread to accurately estimate rotation and scale.

# Eye outer corners
_LEFT_EYE_OUTER: int = 33
_RIGHT_EYE_OUTER: int = 263

# Eye inner corners
_LEFT_EYE_INNER: int = 133
_RIGHT_EYE_INNER: int = 362

# Nose
_NOSE_BRIDGE_TOP: int = 6      # glabella, between the brows
_NOSE_TIP: int = 1

# Mouth corners (fairly stable across expressions for head pose)
_MOUTH_LEFT: int = 61
_MOUTH_RIGHT: int = 291

# Cheekbone / ear-side anchor points
_LEFT_CHEEK: int = 234
_RIGHT_CHEEK: int = 454

# Chin centre
_CHIN: int = 152

# ── Ordered indices used for computing the transform ─────────────────────────
STABLE_LANDMARK_INDICES: List[int] = [
    _LEFT_EYE_OUTER,
    _RIGHT_EYE_OUTER,
    _LEFT_EYE_INNER,
    _RIGHT_EYE_INNER,
    _NOSE_BRIDGE_TOP,
    _NOSE_TIP,
    _MOUTH_LEFT,
    _MOUTH_RIGHT,
    _LEFT_CHEEK,
    _RIGHT_CHEEK,
    _CHIN,
]


# ── Public API ────────────────────────────────────────────────────────────────

def align_face(
    src_image: np.ndarray,
    src_landmarks: List[Tuple[float, float, float]],
    dst_landmarks: List[Tuple[float, float, float]],
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Warp *src_image* so its face aligns geometrically with *dst_landmarks*.

    Computes a 2D **similarity transformation** (rotation + uniform scale +
    translation, no shear) from the source face's stable landmark positions to
    the destination face's stable landmark positions, then applies
    ``cv2.warpAffine`` to the full source image.

    The warped image is the same size as the source image by default (so it
    can be directly used as a drop-in replacement when blending onto the base
    frame), or resized to *output_size* if provided.

    Args:
        src_image:     BGR image array containing the face to be aligned.
        src_landmarks: 478-element list of ``(x, y, z)`` tuples for the source
                       face in absolute image-pixel coordinates.
        dst_landmarks: 478-element list of ``(x, y, z)`` tuples for the
                       destination face (base frame) in absolute pixel coords.
        output_size:   ``(width, height)`` of the output image.  If ``None``,
                       the output size matches *src_image*.

    Returns:
        ``np.ndarray`` — the warped source image aligned to the destination
        face pose, same dtype and channel count as *src_image*.

    Raises:
        ValueError: If landmark lists are too short to contain the required
                    stable anchor indices.
        RuntimeError: If the transform matrix cannot be estimated (e.g. all
                      source points are collinear).
    """
    if len(src_landmarks) < max(STABLE_LANDMARK_INDICES) + 1:
        raise ValueError(
            f"src_landmarks has {len(src_landmarks)} points; need at least "
            f"{max(STABLE_LANDMARK_INDICES) + 1} for the stable anchor set."
        )
    if len(dst_landmarks) < max(STABLE_LANDMARK_INDICES) + 1:
        raise ValueError(
            f"dst_landmarks has {len(dst_landmarks)} points; need at least "
            f"{max(STABLE_LANDMARK_INDICES) + 1} for the stable anchor set."
        )

    # Extract the 2D (x, y) coordinates of the stable points. The z-axis
    # encodes depth perpendicular to the image plane; we drop it here because
    # warpAffine operates in 2D image space.
    src_pts = np.array(
        [(src_landmarks[i][0], src_landmarks[i][1]) for i in STABLE_LANDMARK_INDICES],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [(dst_landmarks[i][0], dst_landmarks[i][1]) for i in STABLE_LANDMARK_INDICES],
        dtype=np.float32,
    )

    # estimateAffinePartial2D finds the *similarity* transform (4 DOF):
    #   scale, rotation, tx, ty — no shear.
    # RANSAC makes it robust to the handful of landmarks that *do* shift
    # slightly with expressions (mouth corners move more than eye corners).
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,  # pixels; outlier threshold
        maxIters=2000,
        confidence=0.99,
    )

    if M is None:
        raise RuntimeError(
            "cv2.estimateAffinePartial2D could not find a valid transform. "
            "This usually means the landmark sets are degenerate (all points "
            "collinear or all at the same location). Check the detector output."
        )

    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    logger.debug(
        "Similarity transform estimated: %d/%d inlier landmarks. "
        "M = [[%.3f, %.3f, %.1f], [%.3f, %.3f, %.1f]]",
        inlier_count, len(STABLE_LANDMARK_INDICES),
        M[0, 0], M[0, 1], M[0, 2],
        M[1, 0], M[1, 1], M[1, 2],
    )

    h, w = src_image.shape[:2]
    out_w, out_h = output_size if output_size else (w, h)

    warped = cv2.warpAffine(
        src_image,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,  # mirrors edges to avoid hard black borders
    )

    logger.info(
        "align_face: warped src (%dx%d) → output (%dx%d) | %d inlier landmarks.",
        w, h, out_w, out_h, inlier_count,
    )
    return warped


def warp_landmarks(
    landmarks: List[Tuple[float, float, float]],
    M: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """Apply affine matrix *M* to a landmark list (keeps z unchanged).

    Utility used when the caller needs the warped landmark positions for
    downstream steps (e.g., blender mask construction from warped coords).

    Args:
        landmarks: List of ``(x, y, z)`` tuples in source image space.
        M:         2×3 affine/similarity matrix (output of
                   ``cv2.estimateAffinePartial2D``).

    Returns:
        List of ``(x', y', z)`` tuples in warped image space.
    """
    pts = np.array([(lm[0], lm[1]) for lm in landmarks], dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    warped_pts = cv2.transform(pts, M).reshape(-1, 2)
    return [
        (float(warped_pts[i, 0]), float(warped_pts[i, 1]), landmarks[i][2])
        for i in range(len(landmarks))
    ]


def compute_similarity_matrix(
    src_landmarks: List[Tuple[float, float, float]],
    dst_landmarks: List[Tuple[float, float, float]],
) -> np.ndarray:
    """Return only the 2×3 similarity transform matrix without applying it.

    Useful when the blender needs the matrix to warp landmark coordinates
    rather than the full image.

    Args:
        src_landmarks: Source face landmarks (``(x, y, z)`` list, ≥478).
        dst_landmarks: Destination face landmarks (``(x, y, z)`` list, ≥478).

    Returns:
        2×3 ``np.float64`` affine matrix.

    Raises:
        RuntimeError: If the matrix cannot be estimated.
    """
    src_pts = np.array(
        [(src_landmarks[i][0], src_landmarks[i][1]) for i in STABLE_LANDMARK_INDICES],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [(dst_landmarks[i][0], dst_landmarks[i][1]) for i in STABLE_LANDMARK_INDICES],
        dtype=np.float32,
    )
    M, _ = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
        confidence=0.99,
    )
    if M is None:
        raise RuntimeError("Could not estimate similarity transform — check landmarks.")
    return M
