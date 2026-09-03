# ==============================================================================
# Production Dockerfile for Classroom CCTV Attendance System
# Optimized for Cloud Deployment (Render, Railway, AWS ECS/EC2, GCP Cloud Run)
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV FLASK_ENV=production
ENV ENVIRONMENT=production

# Install essential system dependencies for OpenCV and InsightFace ONNX runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace buffalo_s models so container starts instantaneously
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))" || true

# Copy project files
COPY . /app/

# Expose HTTP dashboard port
EXPOSE 5000

# Healthcheck for cloud load balancers
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI server using Gunicorn with multi-threading
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "--chdir", "src", "app:app"]
