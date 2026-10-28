# Build:   docker build -t insider-threat-detection .
# Train:   docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/reports:/app/reports insider-threat-detection train
# Serve:   docker run --rm -p 8501:8501 insider-threat-detection serve

# The container supports two commands:
#   train  - runs the training pipeline and writes artefacts to /app/models
#            and /app/reports (mount these to host to persist outputs).
#   serve  - launches the Streamlit prototype on port 8501.

FROM python:3.11-slim

# Avoid bytecode files and stdout buffering for cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed by matplotlib / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source
COPY model/ ./model/
COPY data/ ./data/
COPY app.py .
COPY entrypoint.sh .

# Pre-create output directories
RUN mkdir -p /app/models /app/reports && chmod +x /app/entrypoint.sh

# Streamlit default port
EXPOSE 8501

# Health check for the Streamlit container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
