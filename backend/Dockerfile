# ── Kairos backend Dockerfile ─────────────────────────────────────────────────
#
# Build:   docker build -t kairos-backend .
# Run:     docker run -p 8000:8000 \
#            -e GEMINI_API_KEY=... \
#            -e ANTHROPIC_API_KEY=... \
#            kairos-backend
#
# For demo mode (no API keys needed):
#   docker run -p 8000:8000 -e DEMO_FALLBACK=true kairos-backend
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies required by OpenCV, MediaPipe, and dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir python-multipart

# Copy application source
COPY app/ ./app/

# Create directory for temporary burst uploads (ephemeral, not committed)
RUN mkdir -p temp_bursts

# Expose the FastAPI port
EXPOSE 8000

# Environment variable defaults (override at runtime via -e or Render/Fly dashboard)
ENV DEMO_FALLBACK=false \
    GATE_PROVIDER=gemini \
    GEMINI_MODEL=gemini-1.5-flash \
    ANTHROPIC_MODEL=claude-3-5-sonnet-20240620 \
    PORT=8000

# Start uvicorn — $PORT is honoured by Render automatically
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
