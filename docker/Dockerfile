# Based on https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.7.0-cuda11.0-ubuntu20.04/Dockerfile
FROM nvidia/cuda:10.1-devel-ubuntu18.04

# # Update to newest Nvidia Keyring
# RUN  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
#      && dpkg -i cuda-keyring_1.0-1_all.deb 
User root
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    zip \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app -p
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash usr_nopasswd \
 && chown -R usr_nopasswd:usr_nopasswd /app
RUN echo "usr_nopasswd ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER usr_nopasswd

# All users can use /home/user as their home directory
ENV HOME=/home/usr_nopasswd
RUN chmod 777 /home/usr_nopasswd

# Install Miniconda and Python 3.7
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/usr_nopasswd/miniconda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda clean -ya


#  RUN conda update -n base -c defaults conda 

RUN conda install -y \
        pytorch=1.7.1 \
        torchvision \
        cudatoolkit=10.1 \
        scipy \
        matplotlib \
        jupyter \
        tensorboard \
        configobj  \
        scikit-image  \
        imageio \
        git \
        gitpython \
        pip \    
        h5py \
        pandas \
        parameterized \
        -c pytorch   \
    && pip install imageio_ffmpeg \
    && conda clean -ya 


# Install the application and its Custom Pytorch Operators (Uses read only access tokens)
# If you use VSCode, this will actually generate a duplicate under /app, while the GIT repo  will be mounted  later dynamically.
# This is non-ideal, but allows to pre-build the pytorch operators ahead of time making a first time setup easier. 
RUN cd /app \
    && git clone https://github.com/VLOGroup/LVVCP LVVCP_docker  \
    && cd LVVCP_docker/ \
    && git submodule update --init \
    && REPO_HASH=$(git rev-parse --short HEAD)


# Specify the architecture of your GPU here, a list of common GPUs is setup by default, adapt as needed.
# ENV TORCH_CUDA_ARCH_LIST=Turing
# ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5"

# Install padding Transpose operator
RUN cd /app/LVVCP_docker/Op_pad2d_gh/ \
    && python setup.py install

# Install efficient costvolume sampler
RUN cd /app/LVVCP_docker/Op_sample_cv/ \
    && python setup.py install

# Download pre-trained RAFT Models
RUN cd /app/LVVCP_docker/RAFT_custom/ \
    && ./scripts/download_models.sh \
    && cd /app/LVVCP_docker/




    
#############################################################
# Build this docker image by calling from the git project root directory:
#    nvidia-docker build -t lvvcp docker/
# Run this docker container by calling
#    nvidia-docker run -it --rm --gpus=all lvvcp:latest
# or run it with a folder of images mounted to the containter:
#    nviida-docker run -it --rm --gpus=all -v /path_to_images/:/mnt/path_to_mounted_images lvvcp:latest 

