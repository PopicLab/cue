FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bcftools=1.7-2 \
        build-essential=12.4ubuntu1 \
        libbz2-dev=1.0.6-8.1ubuntu0.2 \
        zlib1g-dev=1:1.2.11.dfsg-0ubuntu2.1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        tabix \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install -U --no-cache-dir \
    setuptools==58.0.0 \
    wheel
RUN pip install -U --no-cache-dir \
    bitarray==1.6.3 \
    cachetools==4.1.0 \
    cython==0.29.21 \
    intervaltree==3.1.0 \
    joblib==0.16.0 \
    matplotlib==3.2.1 \
    numpy==1.18.5 \
    opencv-python==4.5.1.48 \
    pandas==1.0.5 \
    pycocotools==2.0.4 \
    pyfaidx==0.5.9.5 \
    pysam==0.16.0.1 \
    pytabix==0.1 \
    python-dateutil==2.8.1 \
    pyyaml==5.3.1 \
    seaborn==0.11.0 \
    setuptools-scm==6.4.2 \
    torch==1.5.1 \
    torchvision==0.6.1 \
    jupyter

RUN python3 -m pip install Truvari

ENV PYTHONPATH "${PYTHONPATH}:/code"