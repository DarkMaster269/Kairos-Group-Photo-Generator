"""
blender.py — Face mask creation and Poisson blending.

Given an aligned (warped) source image and the destination face's landmark
positions, builds a soft face mask and uses OpenCV's Poisson/Mixed Clone to
blend the source face seamlessly onto the base frame.

Mask pipeline:
  1. Polygon from jawline + eyebrow landmarks → tighter than a convex hull,
     follows the actual face boundary.
  2. Binary fill (cv2.fillPoly).
  3. Morphological erosion to pull the boundary 5–15 px away from ears/hair.
  4. Gaussian blur for feathered soft edge transition.

The blend_face function accepts ``erosion_pixels``, ``feather_radius``, and
``blend_mode`` as tunable arguments so Day 3's retry loop can adjust them
when Gate 2 flags an artifact.

Part of the Kairos CV pipeline: detect → cluster → score → align → **blend** → gate.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from app import config


# ── Face outline landmark indices (MediaPipe 478-point model) ─────────────────
#
# The face silhouette polygon is built from three groups:
#   • Jawline (chin + jaw edges) — bottom/sides of the face.
#   • Left eyebrow — top-left boundary.
#   • Right eyebrow — top-right boundary.
#
# These landmarks trace the visible face boundary closely, so the mask avoids
# including background hair or ears. We deliberately exclude the forehead
# above the brows to prevent the blend from touching hair.
#
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

# Jawline — ordered from left cheek, along chin, to right cheek
JAWLINE_INDICES: List[int] = [
    162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251,
    389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
]

# Left eyebrow — ordered along the brow arch (inner → outer)
LEFT_EYEBROW_INDICES: List[int] = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# Right eyebrow — ordered along the brow arch (inner → outer)
RIGHT_EYEBROW_INDICES: List[int] = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# Forehead padding — nudge the top of the mask slightly above the brows
# so that both eyebrows are fully included (avoids cutting them off).
FOREHEAD_NUDGE_PX: int = 8
"""Number of pixels to extend the eyebrow boundary upward into the forehead."""

# Blend mode constants — exposed for type safety in callers
BLEND_NORMAL: int = cv2.NORMAL_CLONE
"""Standard Poisson blending — best for clean lighting matches."""

BLEND_MIXED: int = cv2.MIXED_CLONE
"""Mixed Poisson blending — better when textures from the base frame should
partially show through (e.g., stubble, freckles)."""


# ── Public API ────────────────────────────────────────────────────────────────

def create_face_mask(
    base_image: np.ndarray,
    landmarks: List[Tuple[float, float, float]],
    erosion_pixels: int = config.DEFAULT_MASK_EROSION,
    feather_radius: int = config.DEFAULT_FEATHER_RADIUS,
) -> np.ndarray:
    """Build a soft face mask from MediaPipe face boundary landmarks.

    Constructs a polygon from the jawline and eyebrow landmark indices,
    fills it as a binary mask, erodes it to pull the boundary away from hair
    and ears, then applies a Gaussian blur to feather the edges for smooth
    Poisson blending transitions.

    Args:
        base_image:     The base / destination BGR image (used only for
                        shape — no pixels are read from it).
        landmarks:      ``(x, y, z)`` landmark list in absolute pixel
                        coordinates of the *destination* face.  Must have
                        at least 478 entries.
        erosion_pixels: Kernel half-size for morphological erosion.
                        Larger values pull the mask boundary further inward,
                        avoiding hair and ear artifacts.
                        Range: typically 3–20.  Default: ``config.DEFAULT_MASK_EROSION``.
        feather_radius: Gaussian blur radius (sigma) for soft edge feathering.
                        Must be a positive odd integer; the function rounds
                        up to the next odd value if needed.
                        Range: typically 5–25.  Default: ``config.DEFAULT_FEATHER_RADIUS``.

    Returns:
        ``np.ndarray`` of shape ``(H, W)`` and dtype ``uint8`` with values
        in ``[0, 255]``.  Pure-white (255) = inside the face region.
        Feathered edges grade from 255 → 0.

    Raises:
        ValueError: If *landmarks* is too short or the polygon is degenerate.
    """
    if len(landmarks) < 477:
        raise ValueError(
            f"Expected ≥ 477 landmarks, got {len(landmarks)}. "
            "Pass the full MediaPipe 478-point landmark list."
        )

    h, w = base_image.shape[:2]

    # 1. Build the face boundary polygon from jawline + eyebrows
    polygon = _build_face_polygon(landmarks, h)
    if len(polygon) < 3:
        raise ValueError("Face boundary polygon is degenerate (fewer than 3 points).")

    # 2. Fill the polygon into a binary uint8 mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=255)

    # 3. Erode — pulls boundary inward away from hair/ears
    if erosion_pixels > 0:
        kernel_size = 2 * erosion_pixels + 1  # must be odd
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.erode(mask, kernel, iterations=1)

    # 4. Gaussian blur — feathers the edges for smooth Poisson blend
    if feather_radius > 0:
        blur_k = _ensure_odd(feather_radius * 2 + 1)
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), sigmaX=feather_radius)

    logger.info(
        "create_face_mask: mask shape=%s, erosion=%dpx, feather=%dpx, "
        "non-zero pixels=%d.",
        mask.shape, erosion_pixels, feather_radius, int(np.count_nonzero(mask)),
    )
    return mask


def blend_face(
    base_image: np.ndarray,
    warped_src_image: np.ndarray,
    mask: np.ndarray,
    blend_mode: int = BLEND_NORMAL,
    center: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Poisson-blend a warped source face onto the base image.

    Uses ``cv2.seamlessClone`` to copy the face region defined by *mask* from
    *warped_src_image* into *base_image* with gradient-domain blending, making
    the skin tones and lighting transition seamlessly.

    The clone center is computed from the mask's weighted centroid (which is
    always inside the filled region) rather than the bounding-box centre, so
    seamlessClone never receives an invalid centre point.

    Args:
        base_image:       Destination BGR image (the base frame composite).
        warped_src_image: Source BGR image already warped to align with
                          *base_image* (output of ``aligner.align_face``).
        mask:             Single-channel uint8 mask (output of
                          ``create_face_mask``).  Non-zero = blend region.
        blend_mode:       ``BLEND_NORMAL`` (``cv2.NORMAL_CLONE``) for standard
                          Poisson blending, or ``BLEND_MIXED``
                          (``cv2.MIXED_CLONE``) to preserve base-frame texture
                          (freckles, stubble) in the blended region.
                          Day 3's retry loop may switch modes when Gate 2
                          flags lighting mismatch artifacts.
        center:           ``(x, y)`` clone center in pixel coordinates.
                          If ``None`` (default), computed automatically from
                          the mask's weighted centroid.

    Returns:
        ``np.ndarray`` — the composite BGR image, same shape as *base_image*.

    Raises:
        ValueError: If images/mask have incompatible shapes or the mask is
                    entirely zero (no region to blend).
    """
    # Validate shapes
    if base_image.shape[:2] != warped_src_image.shape[:2]:
        raise ValueError(
            f"base_image shape {base_image.shape[:2]} ≠ "
            f"warped_src_image shape {warped_src_image.shape[:2]}. "
            "Call aligner.align_face with output_size matching the base frame."
        )
    if mask.shape[:2] != base_image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape[:2]} ≠ base_image shape {base_image.shape[:2]}."
        )
    if not np.any(mask):
        raise ValueError(
            "mask is entirely zero — nothing to blend. "
            "Check that create_face_mask returned a non-empty mask."
        )

    # seamlessClone requires a 3-channel mask
    if mask.ndim == 2:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_3ch = mask

    # Determine clone center
    if center is None:
        center = _mask_centroid(mask)

    # Clamp center to a safe interior position (seamlessClone crashes if
    # the center lies outside the non-zero region)
    center = _clamp_center(center, mask)

    logger.debug(
        "blend_face: center=%s, blend_mode=%s.",
        center,
        "NORMAL_CLONE" if blend_mode == BLEND_NORMAL else "MIXED_CLONE",
    )

    try:
        composite = cv2.seamlessClone(
            src=warped_src_image,
            dst=base_image,
            mask=mask_3ch,
            p=center,
            flags=blend_mode,
        )
    except cv2.error as exc:
        raise RuntimeError(
            f"cv2.seamlessClone failed: {exc}. "
            "Check that center is within the non-zero mask region and that "
            "both images are uint8 BGR."
        ) from exc

    logger.info(
        "blend_face: Poisson blend complete. center=%s, mode=%s.",
        center,
        "NORMAL" if blend_mode == BLEND_NORMAL else "MIXED",
    )
    return composite


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_face_polygon(
    landmarks: List[Tuple[float, float, float]],
    image_height: int,
) -> np.ndarray:
    """Construct a closed face boundary polygon from landmark coordinates.

    Assembles the jawline points (bottom/sides of face) then the right
    eyebrow (top-right), then the left eyebrow reversed (top-left) to form
    a clockwise-wound polygon that covers the visible face region without
    the forehead above the brows.

    Args:
        landmarks:    Full 478-landmark list in absolute pixel coords.
        image_height: Used to nudge eyebrow points slightly upward.

    Returns:
        ``np.ndarray`` of shape ``(N, 1, 2)`` in ``int32`` format (OpenCV
        polygon format).
    """
    # Collect jawline points
    jaw_pts = [(landmarks[i][0], landmarks[i][1]) for i in JAWLINE_INDICES]

    # Collect right eyebrow points
    right_brow = [(landmarks[i][0], landmarks[i][1]) for i in RIGHT_EYEBROW_INDICES]

    # Collect left eyebrow points (reversed so the polygon winds correctly)
    left_brow = [(landmarks[i][0], landmarks[i][1]) for i in reversed(LEFT_EYEBROW_INDICES)]

    # Nudge eyebrow y-coords upward so the mask covers the full brow
    def _nudge_up(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(x, max(0.0, y - FOREHEAD_NUDGE_PX)) for x, y in pts]

    right_brow = _nudge_up(right_brow)
    left_brow = _nudge_up(left_brow)

    # Assemble the full polygon: jaw → right brow → left brow (closing back to jaw)
    all_pts = jaw_pts + right_brow + left_brow

    polygon = np.array(
        [[int(round(x)), int(round(y))] for x, y in all_pts],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    return polygon


def _mask_centroid(mask: np.ndarray) -> Tuple[int, int]:
    """Return the weighted centroid (cx, cy) of the non-zero mask region.

    The centroid is guaranteed to be inside the non-zero region (for a
    convex mask), which is a requirement for ``cv2.seamlessClone``.
    """
    moments = cv2.moments(mask)
    if moments["m00"] < 1.0:
        # Fallback: geometric centre of the bounding rect
        x, y, w, h = cv2.boundingRect(mask)
        return (x + w // 2, y + h // 2)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)


def _clamp_center(
    center: Tuple[int, int],
    mask: np.ndarray,
) -> Tuple[int, int]:
    """Clamp *center* to be inside the mask bounding rect with a 2 px margin.

    seamlessClone raises a cryptic cv2.error if the center is outside the
    non-zero mask region, so we defensively constrain it here.
    """
    x, y, w, h = cv2.boundingRect(mask)
    cx = int(np.clip(center[0], x + 2, x + w - 2))
    cy = int(np.clip(center[1], y + 2, y + h - 2))
    return (cx, cy)


def _ensure_odd(value: int) -> int:
    """Return *value* if it is odd, or *value + 1* if even."""
    return value if value % 2 == 1 else value + 1
