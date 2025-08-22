# Use Python base image instead of CUDA for compatibility
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch CPU version
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY . .

# Install the project and dependencies
RUN pip install --default-timeout=1000 .

# Install Jina
RUN pip install "jina[standard]>=3.11.0"

# Install additional dependencies that might be needed
RUN pip install transformers pillow numpy

# Create non-root user
RUN useradd -m -u 1000 cas && \
    chown -R cas:cas /cas

USER cas

# Expose port
EXPOSE 51000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:51000/ || exit 1

# Start the server
ENTRYPOINT ["python", "-m", "clip_server", "--host", "0.0.0.0", "--port", "51000"]
