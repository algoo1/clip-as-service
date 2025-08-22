ARG CUDA_VERSION=11.6.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ARG CAS_NAME=cas
WORKDIR /${CAS_NAME}

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# constant, wont invalidate cache
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
        curl \
        git \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools nvidia-pyindex \
    && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK
RUN pip install runpod

# Copy the entire project (not just server folder)
COPY . .

# Install clip-server dependencies
RUN pip install --default-timeout=1000 --compile . \
    && pip install jina[standard]

# Create RunPod handler
RUN echo 'import runpod\n\
import os\n\
import subprocess\n\
import time\n\
import signal\n\
import threading\n\
from clip_server.simple_client import SimpleClient\n\
import base64\n\
from io import BytesIO\n\
from PIL import Image\n\
import numpy as np\n\
\n\
# Global variables\n\
clip_process = None\n\
client = None\n\
\n\
def start_clip_server():\n\
    """Start CLIP server in background"""\n\
    global clip_process\n\
    env = os.environ.copy()\n\
    env["CUDA_VISIBLE_DEVICES"] = "0"\n\
    \n\
    clip_process = subprocess.Popen(\n\
        ["python", "-m", "clip_server", "--port", "51000"],\n\
        env=env,\n\
        stdout=subprocess.PIPE,\n\
        stderr=subprocess.PIPE\n\
    )\n\
    \n\
    # Wait for server to be ready\n\
    time.sleep(15)\n\
    print("CLIP server started")\n\
\n\
def init_client():\n\
    """Initialize CLIP client"""\n\
    global client\n\
    if client is None:\n\
        try:\n\
            client = SimpleClient("http://localhost:51000")\n\
            print("CLIP client initialized")\n\
        except Exception as e:\n\
            print(f"Failed to initialize client: {e}")\n\
            return None\n\
    return client\n\
\n\
def handler(job):\n\
    """RunPod handler function"""\n\
    try:\n\
        job_input = job.get("input", {})\n\
        task = job_input.get("task", "encode_image")\n\
        \n\
        # Initialize client if needed\n\
        clip_client = init_client()\n\
        if clip_client is None:\n\
            return {"error": "Failed to initialize CLIP client"}\n\
        \n\
        if task == "encode_image":\n\
            image_data = base64.b64decode(job_input["image"])\n\
            image = Image.open(BytesIO(image_data))\n\
            \n\
            # Convert to numpy array if needed\n\
            img_array = np.array(image)\n\
            \n\
            # Encode image\n\
            embedding = clip_client.encode([img_array])\n\
            \n\
            return {\n\
                "embedding": embedding[0].tolist(),\n\
                "shape": list(embedding[0].shape)\n\
            }\n\
            \n\
        elif task == "encode_text":\n\
            text = job_input["text"]\n\
            \n\
            # Encode text\n\
            embedding = clip_client.encode([text])\n\
            \n\
            return {\n\
                "embedding": embedding[0].tolist(),\n\
                "shape": list(embedding[0].shape)\n\
            }\n\
            \n\
        elif task == "rank":\n\
            image_data = base64.b64decode(job_input["image"])\n\
            image = Image.open(BytesIO(image_data))\n\
            texts = job_input["texts"]\n\
            \n\
            img_array = np.array(image)\n\
            \n\
            # Rank texts by similarity to image\n\
            results = clip_client.rank([img_array], texts)\n\
            \n\
            return {\n\
                "matches": results[0]\n\
            }\n\
            \n\
        else:\n\
            return {"error": f"Unknown task: {task}"}\n\
            \n\
    except Exception as e:\n\
        return {"error": str(e)}\n\
\n\
def cleanup():\n\
    """Cleanup function"""\n\
    global clip_process\n\
    if clip_process:\n\
        clip_process.terminate()\n\
        clip_process.wait()\n\
\n\
if __name__ == "__main__":\n\
    # Start CLIP server\n\
    start_clip_server()\n\
    \n\
    # Register cleanup\n\
    import atexit\n\
    atexit.register(cleanup)\n\
    \n\
    # Start RunPod handler\n\
    runpod.serverless.start({"handler": handler})\n\
' > /cas/runpod_handler.py

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Expose port for CLIP server
EXPOSE 51000

# Use root user for RunPod (serverless needs permissions)
# Remove user creation section from original dockerfile

# Set the entrypoint to our RunPod handler
ENTRYPOINT ["python", "/cas/runpod_handler.py"]
