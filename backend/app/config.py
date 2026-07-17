"""Global Configuration settings for the Kairos pipeline."""

# Expression Scoring Weights
# The sum of these weights should equal 1.0
EYE_WEIGHT: float = 0.4
"""Weight assigned to the eye-openness score (Eyes Aspect Ratio)."""

SMILE_WEIGHT: float = 0.4
"""Weight assigned to the smile/expression score."""

GAZE_WEIGHT: float = 0.2
"""Weight assigned to the gaze-forward score."""


# Face Blending Options
DEFAULT_FEATHER_RADIUS: int = 15
"""Default feathering/blur radius (in pixels) applied to mask edges to soften the blending boundary."""

DEFAULT_MASK_EROSION: int = 5
"""Default number of pixels to erode/shrink the face mask away from boundary edges (like hair or ears)."""


# State Machine Parameters
MAX_RETRIES: int = 2
"""Maximum number of blend retries allowed (swapping faces or widening parameters) before falling back."""


# DeepFace Clustering Parameters
DEEPFACE_MODEL_NAME: str = "ArcFace"
"""The pre-trained face representation model name used by DeepFace (e.g., 'ArcFace', 'Facenet512', 'VGG-Face')."""

CLUSTER_THRESHOLD: float = 0.68
"""Embedding distance threshold (cosine/Euclidean distance) for clustering faces into the same person identity."""


# AI Gateways Configuration
import os

DEMO_FALLBACK: bool = os.environ.get("DEMO_FALLBACK", "False").lower() in ("true", "1", "yes")
"""Whether the AI Gates should use offline cached responses for demonstration purposes."""

GATE_PROVIDER: str = os.environ.get("GATE_PROVIDER", "gemini")
"""Which LLM provider to use for the AI Gates: 'gemini' or 'anthropic'."""

GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
"""The model identifier for Gemini API (e.g., 'gemini-1.5-flash', 'gemini-2.5-flash')."""

ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
"""The model identifier for Anthropic API."""

