# ── Kairos backend Dockerfile (Monorepo Optimized) ───────────────────────────
# Built from the repository root context.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11

WORKDIR /app

# Install system dependencies (libgl1 + X11 libs required by OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir python-multipart

# Copy application source code (includes the already-committed face_landmarker.task)
COPY backend/app/ ./app/

# Pre-download the BlazeFace full-range detector model at build time so the
# first pipeline run doesn't pay that download cost (or risk it failing) live.
# face_landmarker.task is already committed in app/pipeline/, so this only
# needs to fetch the one model that isn't bundled in the repo.
RUN python -c "\
import urllib.request; \
urllib.request.urlretrieve( \
    'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite', \
    'app/pipeline/blaze_face_full_range.tflite'); \
print('BlazeFace full-range detector model pre-downloaded successfully.')"

# Sanity check: make sure both MediaPipe Tasks models actually initialize
# at build time, so a broken model file fails the build, not a live request.
RUN python -c "\
from app.pipeline.detector import FaceMeshDetector; \
d = FaceMeshDetector(); \
d.close(); \
print('FaceMeshDetector (MediaPipe-only) initialized and closed successfully.')"

# Create directory for temporary burst uploads
RUN mkdir -p temp_bursts

EXPOSE 8000

ENV DEMO_FALLBACK=false \
    GATE_PROVIDER=gemini \
    GEMINI_MODEL=gemini-1.5-flash \
    ANTHROPIC_MODEL=claude-3-5-sonnet-20240620 \
    PORT=8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
