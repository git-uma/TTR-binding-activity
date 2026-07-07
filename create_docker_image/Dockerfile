FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

WORKDIR /workspace

RUN rm -f /etc/apt/sources.list.d/cuda.list \
          /etc/apt/sources.list.d/nvidia-ml.list \
          /etc/apt/sources.list.d/*cuda* \
          /etc/apt/sources.list.d/*nvidia* || true

RUN apt-get update && apt-get install -y \
    git \
    fonts-noto-cjk \
    software-properties-common \
    curl \
    gnupg \
    python3.8 \
    python3.8-distutils \
    python3.8-venv \
    build-essential \
    python3.8-dev \
    cmake \
    libopenblas-dev \
    libomp-dev \
    libxrender1 \
    libsm6 \
    libxext6 \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

RUN python3.8 -m pip install --upgrade pip setuptools wheel

RUN python3.8 -m pip install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN python3.8 -m pip install \
    torch-scatter==2.0.9 \
    torch-sparse==0.6.14 \
    torch-cluster==1.5.9 \
    torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

RUN sed -i '/pykan/d' /tmp/requirements.txt && \
    sed -i '/^torch==/d' /tmp/requirements.txt && \
    sed -i '/^torchvision==/d' /tmp/requirements.txt && \
    sed -i '/^torchaudio==/d' /tmp/requirements.txt && \
    sed -i '/^torch-cluster==/d' /tmp/requirements.txt && \
    sed -i '/^torch-scatter==/d' /tmp/requirements.txt && \
    sed -i '/^torch-spline-conv==/d' /tmp/requirements.txt && \
    sed -i '/^torch_sparse==/d' /tmp/requirements.txt && \
    sed -i '/^torch-sparse==/d' /tmp/requirements.txt

RUN python3.8 -m pip install -r /tmp/requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]