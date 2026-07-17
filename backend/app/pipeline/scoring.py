"""
scoring.py — Expression scoring from MediaPipe Face Mesh landmarks.

Computes three independent metrics per face — eyes-open (EAR), smile
curvature, and gaze-forward alignment — then combines them into a single
weighted composite score used to rank face instances within each person
cluster.

All distance calculations use 3D Euclidean distance between landmark
coordinates, making every metric invariant to head roll/tilt.

Part of the Kairos CV pipeline: detect → cluster → **score** → blend → gate.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

from app import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe Face Mesh landmark indices (named constants — no magic numbers)
# ══════════════════════════════════════════════════════════════════════════════

# ── Eye Aspect Ratio (EAR) landmarks ─────────────────────────────────────
# Per-eye 6-point layout for the formula:
#   EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
# p1/p4 = horizontal corners, p2/p6 = outer vertical, p3/p5 = inner vertical.

LEFT_EYE_P1: int = 33    # outer (lateral) corner
LEFT_EYE_P2: int = 160   # upper lid, lateral portion
LEFT_EYE_P3: int = 158   # upper lid, medial portion
LEFT_EYE_P4: int = 133   # inner (medial) corner
LEFT_EYE_P5: int = 153   # lower lid, medial portion
LEFT_EYE_P6: int = 144   # lower lid, lateral portion

RIGHT_EYE_P1: int = 362  # inner (medial) corner
RIGHT_EYE_P2: int = 385  # upper lid, medial portion
RIGHT_EYE_P3: int = 387  # upper lid, lateral portion
RIGHT_EYE_P4: int = 263  # outer (lateral) corner
RIGHT_EYE_P5: int = 373  # lower lid, lateral portion
RIGHT_EYE_P6: int = 380  # lower lid, medial portion

# ── Smile / mouth landmarks ──────────────────────────────────────────────
MOUTH_LEFT_CORNER: int = 61
MOUTH_RIGHT_CORNER: int = 291
UPPER_LIP_CENTER: int = 13
LOWER_LIP_CENTER: int = 14

# Face-width reference (outer eye corners) for normalising mouth width.
FACE_LEFT_REF: int = 33   # left outer eye corner
FACE_RIGHT_REF: int = 263  # right outer eye corner

# Nose bridge — used to derive the face's local "up" direction so that
# mouth-corner elevation is robust to head tilt/roll.
NOSE_BRIDGE_TOP: int = 6   # glabella / between the brows
NOSE_TIP: int = 1          # nose tip

# ── Gaze landmarks ───────────────────────────────────────────────────────
# Iris centres (available when refine_landmarks=True → 478-landmark model).
LEFT_IRIS_CENTER: int = 468
RIGHT_IRIS_CENTER: int = 473

# Eye corners re-used for gaze computation.
LEFT_EYE_INNER: int = LEFT_EYE_P4   # 133
LEFT_EYE_OUTER: int = LEFT_EYE_P1   # 33
RIGHT_EYE_INNER: int = RIGHT_EYE_P1  # 362
RIGHT_EYE_OUTER: int = RIGHT_EYE_P4  # 263

# ── Smile calibration thresholds ─────────────────────────────────────────
# Mouth-width / face-width ratio observed in neutral vs broad-smile faces.
MOUTH_NEUTRAL_RATIO: float = 0.35
MOUTH_SMILE_RATIO: float = 0.65

# EAR calibration — typical range across normal blinking.
EAR_CLOSED: float = 0.10
EAR_OPEN: float = 0.35


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExpressionScore:
    """Per-face expression quality scores.

    All fields are in ``[0.0, 1.0]`` where 1.0 is optimal.

    Attributes:
        eyes_open:       Average EAR-derived openness of both eyes.
        smile:           Mouth-width + corner-elevation smile estimate.
        gaze_forward:    How centred the irises are within the eye opening.
        composite_score: Weighted combination of the above three.
    """

    eyes_open: float
    smile: float
    gaze_forward: float
    composite_score: float


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════

Landmark = Tuple[float, float, float]


def _dist3d(a: Landmark, b: Landmark) -> float:
    """3D Euclidean distance — rotation-invariant."""
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Return unit-length version of *v*, or the zero vector if length ≈ 0."""
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Landmark, b: Landmark) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _midpoint(a: Landmark, b: Landmark) -> Landmark:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


# ── Metric: Eyes Open (EAR) ─────────────────────────────────────────────

def _compute_single_ear(
    p1: Landmark,
    p2: Landmark,
    p3: Landmark,
    p4: Landmark,
    p5: Landmark,
    p6: Landmark,
) -> float:
    """Eye Aspect Ratio for one eye.

    EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
    """
    horizontal = _dist3d(p1, p4)
    if horizontal < 1e-9:
        return 0.0
    vertical = _dist3d(p2, p6) + _dist3d(p3, p5)
    return vertical / (2.0 * horizontal)


def _compute_eyes_open(lms: List[Landmark]) -> float:
    """Average EAR of both eyes, normalised to [0, 1].

    Uses 3D distances so the value is stable across head tilt angles.
    """
    left_ear = _compute_single_ear(
        lms[LEFT_EYE_P1], lms[LEFT_EYE_P2], lms[LEFT_EYE_P3],
        lms[LEFT_EYE_P4], lms[LEFT_EYE_P5], lms[LEFT_EYE_P6],
    )
    right_ear = _compute_single_ear(
        lms[RIGHT_EYE_P1], lms[RIGHT_EYE_P2], lms[RIGHT_EYE_P3],
        lms[RIGHT_EYE_P4], lms[RIGHT_EYE_P5], lms[RIGHT_EYE_P6],
    )
    avg_ear = (left_ear + right_ear) / 2.0

    # Map the raw EAR range into [0, 1].
    return _clamp((avg_ear - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED))


# ── Metric: Smile ────────────────────────────────────────────────────────

def _compute_smile(lms: List[Landmark]) -> float:
    """Estimate smile intensity from mouth width and corner elevation.

    Two signals are combined:
    1. **Width ratio** — mouth width ÷ face width.  Smiling stretches the
       mouth wider.  Both distances use 3D coordinates, so the ratio is
       invariant to scale and rotation.
    2. **Corner elevation** — how much the mouth corners are raised in the
       face's own "up" direction (derived from the nose bridge vector).
       Positive = corners pulled up = smiling.  Also rotation-invariant
       because we project onto the face-local vertical axis.

    Returns a float in [0, 1].
    """
    face_width = _dist3d(lms[FACE_LEFT_REF], lms[FACE_RIGHT_REF])
    if face_width < 1e-9:
        return 0.0

    mouth_width = _dist3d(lms[MOUTH_LEFT_CORNER], lms[MOUTH_RIGHT_CORNER])
    width_ratio = mouth_width / face_width
    width_score = _clamp(
        (width_ratio - MOUTH_NEUTRAL_RATIO)
        / (MOUTH_SMILE_RATIO - MOUTH_NEUTRAL_RATIO)
    )

    # ── Corner elevation (head-tilt robust) ──────────────────────────
    # "Up" on the face = direction from nose tip toward the brow bridge.
    face_up = _normalize(_sub(lms[NOSE_BRIDGE_TOP], lms[NOSE_TIP]))
    lip_center = _midpoint(lms[UPPER_LIP_CENTER], lms[LOWER_LIP_CENTER])
    corner_mid = _midpoint(lms[MOUTH_LEFT_CORNER], lms[MOUTH_RIGHT_CORNER])
    displacement = _sub(corner_mid, lip_center)
    elevation = _dot(displacement, face_up)

    # Normalise elevation by face width so it's scale-independent.
    norm_elev = elevation / face_width
    # Empirical range: ~-0.02 (frown) to ~+0.04 (broad smile).
    elevation_score = _clamp((norm_elev + 0.02) / 0.06)

    return _clamp(0.6 * width_score + 0.4 * elevation_score)


# ── Metric: Gaze Forward ─────────────────────────────────────────────────

def _gaze_single_eye(
    iris_center: Landmark,
    inner_corner: Landmark,
    outer_corner: Landmark,
) -> float:
    """How centred the iris is between the eye corners (one eye).

    Projects the iris position onto the inner→outer axis and returns
    1.0 when perfectly centred, dropping toward 0.0 as the iris drifts
    to either extreme.  Uses 3D distances for head-tilt robustness.

    Returns a float in [0, 1].
    """
    eye_width = _dist3d(inner_corner, outer_corner)
    if eye_width < 1e-9:
        return 0.5

    dist_inner = _dist3d(iris_center, inner_corner)
    ratio = dist_inner / eye_width  # 0.5 = perfectly centred

    return _clamp(1.0 - 2.0 * abs(ratio - 0.5))


def _compute_gaze_forward(lms: List[Landmark]) -> float:
    """Average gaze-forward score for both eyes.

    Falls back to 0.5 (neutral) if iris landmarks are unavailable
    (i.e. the detector was run with ``refine_landmarks=False``, giving
    only 468 points instead of 478).
    """
    if len(lms) < 474:
        # No iris landmarks available — return neutral score.
        logger.debug(
            "Only %d landmarks (< 474) — iris unavailable, returning 0.5.",
            len(lms),
        )
        return 0.5

    left = _gaze_single_eye(
        lms[LEFT_IRIS_CENTER], lms[LEFT_EYE_INNER], lms[LEFT_EYE_OUTER]
    )
    right = _gaze_single_eye(
        lms[RIGHT_IRIS_CENTER], lms[RIGHT_EYE_INNER], lms[RIGHT_EYE_OUTER]
    )
    return (left + right) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def score_face(landmarks: List[Tuple[float, float, float]]) -> ExpressionScore:
    """Score a single face's expression quality.

    Computes three independent metrics (eyes-open, smile, gaze-forward)
    from MediaPipe Face Mesh 3D landmarks and combines them into a
    weighted composite score using the weights in ``config.py``.

    All internal distance calculations use 3D Euclidean distance, so
    results are stable across head tilt and roll angles.

    Args:
        landmarks: List of ``(x, y, z)`` tuples — 468 or 478 entries
                   (output of ``FaceMeshDetector.detect_faces``).

    Returns:
        An :class:`ExpressionScore` instance with all fields in [0, 1].

    Raises:
        ValueError: If *landmarks* has fewer than 468 entries.
    """
    if len(landmarks) < 468:
        raise ValueError(
            f"Expected ≥ 468 landmarks, got {len(landmarks)}. "
            "Pass the full MediaPipe Face Mesh output."
        )

    eyes = _compute_eyes_open(landmarks)
    smile = _compute_smile(landmarks)
    gaze = _compute_gaze_forward(landmarks)

    composite = (
        config.EYE_WEIGHT * eyes
        + config.SMILE_WEIGHT * smile
        + config.GAZE_WEIGHT * gaze
    )

    score = ExpressionScore(
        eyes_open=round(eyes, 4),
        smile=round(smile, 4),
        gaze_forward=round(gaze, 4),
        composite_score=round(composite, 4),
    )

    logger.debug(
        "Scored face: eyes=%.2f smile=%.2f gaze=%.2f → composite=%.2f",
        eyes, smile, gaze, composite,
    )
    return score
