FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.7
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.7 numpy pyyaml scipy pandas h5py mkl mkl-include ninja cython ipython typing \
 && conda clean -ya \
 && pip install mlflow

# CUDA 10.1-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    pytorch=1.5.1 \
    torchvision \
 && conda clean -ya

RUN git clone https://github.com/nvidia/apex /apex \
  && cd /apex \
  && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /workspace
RUN rm -rf apex

# Set the default command to python3
CMD ["python3"]

WORKDIR /workspace
RUN chmod -R a+w .