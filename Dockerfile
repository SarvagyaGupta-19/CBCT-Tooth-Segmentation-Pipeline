FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
LABEL maintainer="Clinical AI Team" description="3D CBCT tooth segmentation pipeline"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip git wget \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

ENV nnUNet_raw=/data/nnunet_raw
ENV nnUNet_preprocessed=/data/nnunet_preprocessed
ENV nnUNet_results=/runs/nnunet_results
RUN mkdir -p ${nnUNet_raw} ${nnUNet_preprocessed} ${nnUNet_results}

COPY . /app
VOLUME ["/data", "/runs", "/results"]

CMD ["python3", "scripts/unet_inference.py", \
     "--input",      "/data/demo/example.nii.gz", \
     "--weights",    "/runs/cbct_seg/best_model.pth", \
     "--output-dir", "/results/demo"]
