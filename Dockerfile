# Use a stable and available CUDA image
FROM nvidia/cuda:11.8-devel-ubuntu20.04

ARG CAS_NAME=cas
WORKDIR /${CAS_NAME}

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.vendor="Jina AI Limited" \
      org.opencontainers.image.licenses="Apache 2.0" \
      org.opencontainers.image.title="CLIP-as-Service" \
      org.opencontainers.image.description="Embed images and sentences into fixed-length vectors with CLIP" \
      org.opencontainers.image.authors="hello@jina.ai" \
      org.opencontainers.image.url="clip-as-service" \
      org.opencontainers.image.documentation="https://clip-as-service.jina.ai/"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        wget \
        git \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools nvidia-pyindex \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.8 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the project files
COPY . .

# Install the project
RUN pip install --default-timeout=1000 --compile .

# Install Jina
RUN pip install "jina[standard]"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Create non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=${CAS_NAME}
ARG GROUP_NAME=${CAS_NAME}

RUN groupadd -g ${GROUP_ID} ${USER_NAME} &&\
    useradd -l -u ${USER_ID} -g ${USER_NAME} ${GROUP_NAME} &&\
    mkdir /home/${USER_NAME} &&\
    chown ${USER_NAME}:${GROUP_NAME} /home/${USER_NAME} &&\
    chown -R ${USER_NAME}:${GROUP_NAME} /${CAS_NAME}/

USER ${USER_NAME}

EXPOSE 51000

ENTRYPOINT ["python", "-m", "clip_server"]
