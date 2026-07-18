# ── Kairos backend Dockerfile (Monorepo Optimized) ───────────────────────────
# Built from the repository root context.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the backend folder relative to the repository root
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir python-multipart

# Copy application source code from the backend folder relative to the repository root
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
