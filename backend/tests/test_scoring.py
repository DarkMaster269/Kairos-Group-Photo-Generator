import pytest
from typing import List, Tuple

from app.pipeline.scoring import (
    score_face,
    LEFT_EYE_P1, LEFT_EYE_P2, LEFT_EYE_P3, LEFT_EYE_P4, LEFT_EYE_P5, LEFT_EYE_P6,
    RIGHT_EYE_P1, RIGHT_EYE_P2, RIGHT_EYE_P3, RIGHT_EYE_P4, RIGHT_EYE_P5, RIGHT_EYE_P6,
    MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER, UPPER_LIP_CENTER, LOWER_LIP_CENTER,
    FACE_LEFT_REF, FACE_RIGHT_REF, NOSE_BRIDGE_TOP, NOSE_TIP,
    LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    LEFT_EYE_INNER, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER
)

# Helper to create a base list of neutral landmarks
def get_base_landmarks() -> List[Tuple[float, float, float]]:
    lms = [(0.0, 0.0, 0.0)] * 478
    
    # Define face width reference (width = 100)
    lms[FACE_LEFT_REF] = (50.0, 100.0, 0.0)
    lms[FACE_RIGHT_REF] = (150.0, 100.0, 0.0)
    
    # Define nose coordinates for face vertical direction
    lms[NOSE_TIP] = (100.0, 150.0, 0.0)
    lms[NOSE_BRIDGE_TOP] = (100.0, 100.0, 0.0) # Up direction is (0, -50, 0) -> normalized (0, -1, 0)
    
    # Define mouth centers
    lms[UPPER_LIP_CENTER] = (100.0, 180.0, 0.0)
    lms[LOWER_LIP_CENTER] = (100.0, 185.0, 0.0)
    
    # Default mouth corners (Neutral width ratio = 40 / 100 = 0.40)
    lms[MOUTH_LEFT_CORNER] = (80.0, 182.5, 0.0)
    lms[MOUTH_RIGHT_CORNER] = (120.0, 182.5, 0.0)
    
    # Define horizontal corners for both eyes (width = 10)
    lms[LEFT_EYE_P1] = (70.0, 100.0, 0.0)  # outer
    lms[LEFT_EYE_P4] = (80.0, 100.0, 0.0)  # inner
    lms[RIGHT_EYE_P1] = (120.0, 100.0, 0.0) # inner
    lms[RIGHT_EYE_P4] = (130.0, 100.0, 0.0) # outer
    
    # Set default eye lids
    lms[LEFT_EYE_P2] = (73.0, 98.0, 0.0)
    lms[LEFT_EYE_P3] = (77.0, 98.0, 0.0)
    lms[LEFT_EYE_P5] = (77.0, 102.0, 0.0)
    lms[LEFT_EYE_P6] = (73.0, 102.0, 0.0)
    
    lms[RIGHT_EYE_P2] = (123.0, 98.0, 0.0)
    lms[RIGHT_EYE_P3] = (127.0, 98.0, 0.0)
    lms[RIGHT_EYE_P5] = (127.0, 102.0, 0.0)
    lms[RIGHT_EYE_P6] = (123.0, 102.0, 0.0)
    
    # Set default iris centers (centered)
    lms[LEFT_IRIS_CENTER] = (75.0, 100.0, 0.0)
    lms[RIGHT_IRIS_CENTER] = (125.0, 100.0, 0.0)
    
    return lms

def test_score_face_insufficient_landmarks():
    """Verify that score_face raises ValueError when passed too few landmarks."""
    invalid_landmarks = [(0.0, 0.0, 0.0)] * 400
    with pytest.raises(ValueError, match="Expected ≥ 468 landmarks"):
        score_face(invalid_landmarks)

def test_eyes_aspect_ratio_open():
    """Verify eyes openness score matches fully open when vertically spaced."""
    lms = get_base_landmarks()
    # Left eye vertical distance: p2-p6 = 8, p3-p5 = 8. Horizontal = 10.
    # Raw EAR = (8 + 8) / 20 = 0.80.
    # Since EAR_OPEN = 0.35, this is well above open threshold and should yield 1.0.
    lms[LEFT_EYE_P2] = (73.0, 96.0, 0.0)
    lms[LEFT_EYE_P3] = (77.0, 96.0, 0.0)
    lms[LEFT_EYE_P5] = (77.0, 104.0, 0.0)
    lms[LEFT_EYE_P6] = (73.0, 104.0, 0.0)
    
    lms[RIGHT_EYE_P2] = (123.0, 96.0, 0.0)
    lms[RIGHT_EYE_P3] = (127.0, 96.0, 0.0)
    lms[RIGHT_EYE_P5] = (127.0, 104.0, 0.0)
    lms[RIGHT_EYE_P6] = (123.0, 104.0, 0.0)
    
    scores = score_face(lms)
    assert scores.eyes_open == 1.0

def test_eyes_aspect_ratio_closed():
    """Verify eyes openness score matches 0.0 when lids are closed (aligned horizontally)."""
    lms = get_base_landmarks()
    # Left and right eye vertical points very close (closed eye)
    for p in [LEFT_EYE_P2, LEFT_EYE_P3, LEFT_EYE_P5, LEFT_EYE_P6]:
        lms[p] = (lms[p][0], 100.0, 0.0)
    for p in [RIGHT_EYE_P2, RIGHT_EYE_P3, RIGHT_EYE_P5, RIGHT_EYE_P6]:
        lms[p] = (lms[p][0], 100.0, 0.0)
        
    scores = score_face(lms)
    assert scores.eyes_open == 0.0

def test_smile_neutral_vs_broad():
    """Verify that a wider, upturned mouth yields a higher smile score than a narrow/neutral mouth."""
    # 1. Neutral mouth (narrower width, flat corners)
    lms_neutral = get_base_landmarks()
    lms_neutral[MOUTH_LEFT_CORNER] = (85.0, 182.5, 0.0)
    lms_neutral[MOUTH_RIGHT_CORNER] = (115.0, 182.5, 0.0)
    
    # 2. Smiling mouth (stretched wide, corners pulled UP towards nose bridge top)
    lms_smiling = get_base_landmarks()
    # Stretched width (width = 135 - 65 = 70. Width ratio = 0.70)
    lms_smiling[MOUTH_LEFT_CORNER] = (65.0, 175.0, 0.0)
    lms_smiling[MOUTH_RIGHT_CORNER] = (135.0, 175.0, 0.0)
    
    score_neutral = score_face(lms_neutral)
    score_smiling = score_face(lms_smiling)
    
    assert score_smiling.smile > score_neutral.smile
    assert score_smiling.smile > 0.8
    assert score_neutral.smile < 0.5

def test_gaze_direction():
    """Verify gaze centered yields 1.0, while iris drifted to corners drops the score."""
    # Centered gaze
    lms_centered = get_base_landmarks()
    score_centered = score_face(lms_centered)
    assert score_centered.gaze_forward == 1.0
    
    # Left iris shifted to the left outer corner (70.0)
    lms_drifted = get_base_landmarks()
    lms_drifted[LEFT_IRIS_CENTER] = (70.5, 100.0, 0.0)
    
    score_drifted = score_face(lms_drifted)
    assert score_drifted.gaze_forward < 0.6
