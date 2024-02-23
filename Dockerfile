# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

# Set the working directory in the container to /mobi
WORKDIR /mobi

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    g++ \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libgtk2.0-dev \
    git \
    openssh-server

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Clone the repository
RUN git clone https://github.com/Fantasy-Studio/Paint-by-Example.git
WORKDIR /mobi/Paint-by-Example

# Create a new conda environment from the environment.yaml file
RUN conda env create -f environment.yaml
RUN conda init bash

SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate Paint-by-Example
