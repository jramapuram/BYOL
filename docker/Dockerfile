FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    htop \
    git \
    curl \
    imagemagick \
    emacs-nox \
    ca-certificates \
    rsync \
    gtk-doc-tools \
    autotools-dev \
    automake \
    libgl1-mesa-glx \
    swig && \
    rm -rf /var/lib/apt/lists/*

# install base conda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Anaconda_37.sh && \
    sh ./Anaconda*.sh -b -p /opt/conda && rm ./Anaconda*.sh
ENV PATH /opt/conda/bin:$PATH

# install conda deps
RUN /opt/conda/bin/conda install numpy pandas scipy scikit-image scikit-learn h5py matplotlib && \
    /opt/conda/bin/conda install pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch && \
    /opt/conda/bin/conda install -y -c conda-forge opencv pyvips

# install misc conda / pip deps
# consider installing jax jaxlib tensorflow-gpu==1.14
RUN /opt/conda/bin/pip install torchvision tqdm msgpack joblib visdom rpyc \
    pyro-ppl tensorboardX psutil dm-tree tensorflow==1.14 && \
    /opt/conda/bin/conda install -c defaults libprotobuf protobuf

# install apex & cleanup
RUN git clone --recursive https://github.com/NVIDIA/apex /opt/apex && \
    cd /opt/apex && /opt/conda/bin/pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /workspace
RUN chmod -R a+w /workspace #&& chmod -R a+rwx /opt/conda
