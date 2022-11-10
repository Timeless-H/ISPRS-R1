ARG IMAGE_NAME
FROM ${IMAGE_NAME}:11.6.0-runtime-ubuntu20.04 as base

ENV NV_CUDA_LIB_VERSION "11.6.0-1"

FROM base as base-amd64

ENV NV_CUDA_CUDART_DEV_VERSION 11.6.55-1
ENV NV_NVML_DEV_VERSION 11.6.55-1
ENV NV_LIBCUSPARSE_DEV_VERSION 11.7.1.55-1
ENV NV_LIBNPP_DEV_VERSION 11.6.0.55-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-11-6=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 11.8.1.74-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-11-6
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_NVPROF_VERSION 11.6.55-1
ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-11-6=${NV_NVPROF_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.12.10-1
ENV NCCL_VERSION 2.12.10-1
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.6
FROM base as base-arm64

ENV NV_CUDA_CUDART_DEV_VERSION 11.6.55-1
ENV NV_NVML_DEV_VERSION 11.6.55-1
ENV NV_LIBCUSPARSE_DEV_VERSION 11.7.1.55-1
ENV NV_LIBNPP_DEV_VERSION 11.6.0.55-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-11-6=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-11-6
ENV NV_LIBCUBLAS_DEV_VERSION 11.8.1.74-1
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_NVPROF_VERSION 11.6.55-1
ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-11-6=${NV_NVPROF_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.12.10-1
ENV NCCL_VERSION 2.12.10-1
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.6

ARG TARGETARCH
FROM base-amd64

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-11-6=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-11-6=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-11-6=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-11-6=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-11-6=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-11-6=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

#--upgrade
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install" && \
    GIT_CLONE="git clone --depth 10" && \
    apt-get update && apt-get install -y --no-install-recommends &&\
     DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
	    nano \
        libx11-dev \
        fish \
        libsparsehash-dev \
        sqlite3 \
        libsqlite3-dev \
        curl \
        libcurl4-openssl-dev \
        # python3-opengl \
        pkg-config \
        && \
        DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
    && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        # python3-pip \
        # python-wheel \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
        $PIP_INSTALL \
        setuptools \ 
        numpy \
        scipy \
        matplotlib \
        Cython \
        tqdm \
        provider \
        imageio \
        joblib \
        tensorboard \
        ninja \
        memcnn \
        scikit-image \
        jupyter \
        sklearn \
        # numba \
        einops \
        # opencv-python \
        # open3d \
        torchsummary \
        h5py \
        # hdf5storage \
        pandas \
        PyYAML \
        Pillow \
        plyfile \
        # pyntcloud \
        pickleshare \
        # trimesh \
        # pyrender \
        # mesh-to-sdf \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL \
    torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html 
    # torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 

RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# RUN GIT_CLONE="git clone --recurse-submodules" && \
#     $GIT_CLONE \
#         https://github.com/unlimblue/KNN_CUDA.git && \
#     cd KNN_CUDA && make && make install && cd && \
#     wget -P /usr/bin https://github.com/unlimblue/KNN_CUDA/raw/master/ninja

RUN chmod -R 777 /usr/local/lib/python3.8/dist-packages/knn_cuda

# workdir is where u work in dockers
# copy . /app copies content of ur supposed working dir to the docker wk dir

WORKDIR /app
COPY . /app

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser


RUN python -c "import torch; print(torch.__version__)" 
RUN python -c "import torch; print(torch.version.cuda)" 
RUN python -c "import torch; print(torch.cuda.is_available())" 

CMD ["python", "main.py "]
