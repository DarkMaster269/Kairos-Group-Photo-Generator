# ── Kairos backend Dockerfile (Railway & Render optimized) ───────────────────
# Uses the full Python image (which contains build-essential, cmake, and standard
# C/C++ libraries pre-installed) to bypass the runc/apt-get build container crash on Railway.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies that might not be in the full image
# (Headless OpenCV is used, which minimizes GL dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir python-multipart

# Copy application source
COPY app/ ./app/

# Create directory for temporary burst uploads (ephemeral)
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
