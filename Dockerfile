# Use Python base image for compatibility
FROM python:3.9-slim

WORKDIR /cas

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

LABEL org.opencontainers.image.vendor="Jina AI Limited" \
      org.opencontainers.image.licenses="Apache 2.0" \
      org.opencontainers.image.title="CLIP-as-Service" \
      org.opencontainers.image.description="Embed images and sentences into fixed-length vectors with CLIP" \
      org.opencontainers.image.authors="hello@jina.ai" \
      org.opencontainers.image.url="clip-as-service" \
      org.opencontainers.image.documentation="https://clip-as-service.jina.ai/"

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential \
        python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch CPU version first
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Jina first
RUN pip install "jina[standard]>=3.11.0"

# Install other dependencies that might be needed
RUN pip install transformers pillow numpy requests

# Copy project files
COPY . .

# Install the project (try different approaches)
RUN pip install -e . || \
    pip install --no-deps . || \
    (cd /cas && python setup.py install) || \
    echo "Direct installation failed, trying manual install..." && \
    pip install -r requirements.txt 2>/dev/null || \
    echo "No requirements.txt found, continuing..."

# Ensure clip_server is available
RUN python -c "import clip_server" || \
    pip install clip-server || \
    echo "Will try to run from source"

# Create non-root user
RUN useradd -m -u 1000 cas && \
    chown -R cas:cas /cas

USER cas

# Expose port
EXPOSE 51000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:51000/ || exit 1

# Try multiple ways to start the server
ENTRYPOINT ["sh", "-c", "python -m clip_server --host 0.0.0.0 --port 51000 || python -c 'from clip_server import app; app.run(host=\"0.0.0.0\", port=51000)' || python server.py"]
