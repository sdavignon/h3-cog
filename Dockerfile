FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_ROOT=/opt/ComfyUI

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      python3 \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
      nvidia-cudnn-cu12==9.19.0.56 \
      nvidia-cublas-cu12==12.8.4.1 \
      nvidia-cusparselt-cu12==0.7.1

RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
      nvidia-nccl-cu12==2.28.9 \
      nvidia-nvshmem-cu12==3.4.5 \
      nvidia-cufft-cu12==11.3.3.83 \
      nvidia-cusolver-cu12==11.7.3.90 \
      nvidia-cusparse-cu12==12.5.8.93

RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
      nvidia-cuda-runtime-cu12==12.8.90 \
      nvidia-cuda-cupti-cu12==12.8.90 \
      nvidia-cufile-cu12==1.13.1.3 \
      nvidia-curand-cu12==10.3.9.90 \
      nvidia-nvjitlink-cu12==12.8.93 \
      nvidia-cuda-nvrtc-cu12==12.8.93 \
      nvidia-nvtx-cu12==12.8.90

RUN python3 -m pip install --break-system-packages \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      torch==2.11.0 \
      torchvision==0.26.0 \
      torchaudio==2.11.0 \
      requests==2.32.5 \
      runpod==1.7.13 \
      boto3==1.40.12

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_ROOT" \
    && git -C "$COMFY_ROOT" checkout 9a9fdb10ed144ce760d9682cb247526ea23cc525 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
      -r "$COMFY_ROOT/requirements.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu128

WORKDIR /src
COPY . /src

CMD ["python3", "-u", "/src/rp_handler.py"]
