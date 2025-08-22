ARG CUDA_VERSION=11.6.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

WORKDIR /cas

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

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

# Copy the entire project
COPY . .

# Install clip-server dependencies
RUN pip install --default-timeout=1000 --compile . \
    && pip install jina[standard]

# Create RunPod handler file
COPY <<EOF /cas/runpod_handler.py
import runpod
import os
import subprocess
import time
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import json

# Global variables
clip_process = None

def start_clip_server():
    """Start CLIP server in background"""
    global clip_process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    clip_process = subprocess.Popen(
        ["python", "-m", "clip_server", "--host", "0.0.0.0", "--port", "51000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to be ready
    print("Starting CLIP server...")
    time.sleep(20)
    
    # Test connection
    for i in range(10):
        try:
            response = requests.get("http://localhost:51000/", timeout=5)
            if response.status_code == 200:
                print("CLIP server is ready!")
                return True
        except:
            time.sleep(2)
    
    print("CLIP server failed to start properly")
    return False

def make_clip_request(endpoint, data):
    """Make request to CLIP server"""
    try:
        response = requests.post(
            f"http://localhost:51000/{endpoint}",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job.get("input", {})
        task = job_input.get("task", "encode")
        
        if task == "encode":
            # Handle both image and text encoding
            if "image" in job_input:
                # Decode base64 image
                image_data = base64.b64decode(job_input["image"])
                
                # Save image temporarily
                with open("/tmp/temp_image.jpg", "wb") as f:
                    f.write(image_data)
                
                # Make request to CLIP server
                with open("/tmp/temp_image.jpg", "rb") as f:
                    files = {"data": f}
                    response = requests.post(
                        "http://localhost:51000/encode",
                        files=files,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "embedding": result.get("data", []),
                        "shape": result.get("shape", [])
                    }
                else:
                    return {"error": f"CLIP server error: {response.text}"}
                    
            elif "text" in job_input:
                # Text encoding
                data = {"data": job_input["text"]}
                result = make_clip_request("encode", data)
                
                if "error" not in result:
                    return {
                        "embedding": result.get("data", []),
                        "shape": result.get("shape", [])
                    }
                else:
                    return result
        
        elif task == "rank":
            # Image-text ranking
            if "image" in job_input and "texts" in job_input:
                image_data = base64.b64decode(job_input["image"])
                
                with open("/tmp/temp_image.jpg", "wb") as f:
                    f.write(image_data)
                
                # Prepare data for ranking
                data = {
                    "data": job_input["texts"],
                    "image": "/tmp/temp_image.jpg"
                }
                
                with open("/tmp/temp_image.jpg", "rb") as f:
                    files = {"data": (None, json.dumps(data["data"])), "image": f}
                    response = requests.post(
                        "http://localhost:51000/rank",
                        files=files,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"CLIP server error: {response.text}"}
        
        return {"error": f"Unknown task: {task}"}
        
    except Exception as e:
        return {"error": str(e)}

def cleanup():
    """Cleanup function"""
    global clip_process
    if clip_process:
        clip_process.terminate()
        clip_process.wait()

if __name__ == "__main__":
    # Start CLIP server
    if start_clip_server():
        # Register cleanup
        import atexit
        atexit.register(cleanup)
        
        # Start RunPod handler
        print("Starting RunPod handler...")
        runpod.serverless.start({"handler": handler})
    else:
        print("Failed to start CLIP server. Exiting.")
        exit(1)
EOF

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

EXPOSE 51000

ENTRYPOINT ["python", "/cas/runpod_handler.py"]
