# ── Kairos backend Dockerfile (Monorepo Optimized) ───────────────────────────
# Built from the repository root context.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11

# Set working directory inside the container
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

# Pre-download MediaPipe FaceMesh model files at build time so the first
# pipeline run does not hang waiting for a runtime download from Google.
RUN python -c "\
import mediapipe as mp; \
detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, refine_landmarks=True); \
print('MediaPipe FaceMesh model pre-loaded successfully.'); \
detector.close()"

# Copy application source code
COPY backend/app/ ./app/

# Create directory for temporary burst uploads
RUN mkdir -p temp_bursts

# Expose the FastAPI port
EXPOSE 8000

# Environment variable defaults
ENV DEMO_FALLBACK=false \
    GATE_PROVIDER=gemini \
    GEMINI_MODEL=gemini-1.5-flash \
    ANTHROPIC_MODEL=claude-3-5-sonnet-20240620 \
    PORT=8000

# Start uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
