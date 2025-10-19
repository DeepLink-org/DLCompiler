FROM ccr.ccs.tencentyun.com/library/ubuntu:22.04

# FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"

RUN apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    bash \
    ca-certificates \
    curl \
    git \
    gnupg \
    make \
    sudo \
    unzip \
    vim \
    wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 && \
    apt-get update

RUN apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    clang \
    clang-format \
    cmake \
    lld \
    ninja-build \
    python3-dev \
    python3-venv \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
      --slave /usr/bin/python3 python3 /usr/bin/python3.9 \
      --slave /usr/bin/pip pip /usr/local/bin/pip3.9 \
      --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.9 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2 \
      --slave /usr/bin/python3 python3 /usr/bin/python3.10 \
      --slave /usr/bin/pip pip /usr/local/bin/pip3.10 \
      --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 3 \
      --slave /usr/bin/python3 python3 /usr/bin/python3.11 \
      --slave /usr/bin/pip pip /usr/local/bin/pip3.11 \
      --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.11 && \
    rm -f /usr/local/bin/pip /usr/local/bin/pip3 && \
    update-alternatives --set python /usr/bin/python3.10

USER root
WORKDIR /root/workspace

RUN set -ex && \
    apt update && \
    apt install -y zlib1g-dev clang-15 lld-15 && \
    apt install -y ccache && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 20

# 添加Miniconda安装（Python 3.10版本）
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-aarch64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -f /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -tipy && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate base" >> /etc/bash.bashrc

ENV PATH $CONDA_DIR/bin:$PATH

# 初始化 conda
RUN conda init bash

# 创建新的 conda 环境
RUN set -ex \
    && conda create -n dlcompiler python=3.10
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/lib:/lib:$CONDA_DIR/envs/dlcompiler/lib:$LD_LIBRARY_PATH

RUN echo "source activate dlcompiler" > ~/.bashrc
RUN set -ex && \
    apt-get update && apt-get install -y openssh-server && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    service ssh start

COPY cann cann
RUN set -ex && \
    cd /root/workspace/cann && \
    chmod +x Ascend-cann-toolkit_8.3.RC1.alpha003_linux-aarch64.run && \
    chmod +x Ascend-cann-kernels-910b_8.3.RC1.alpha003_linux-aarch64.run && \
    ./Ascend-cann-toolkit_8.3.RC1.alpha003_linux-aarch64.run --quiet --install --install-for-all && \
    ./Ascend-cann-kernels-910b_8.3.RC1.alpha003_linux-aarch64.run --quiet --install --install-for-all && \
    echo "alias ll='ls -alh'" >> ~/.bashrc && \
    source ~/.bashrc && \
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc

RUN set -ex && \
    source ~/.bashrc && \
    conda activate dlcompiler && \
    pip install ninja cmake wheel pybind11  -i https://mirrors.huaweicloud.com/repository/pypi/simple    && \
    pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml torch==2.6.0 torch-npu==2.6.0  -i https://mirrors.huaweicloud.com/repository/pypi/simple

COPY requirements.txt .
RUN set -ex && \
    source ~/.bashrc && \
    conda activate dlcompiler && \
    pip install -r requirements.txt  -i https://mirrors.huaweicloud.com/repository/pypi/simple

ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

RUN mkdir -p /root/.triton
COPY nvidia /root/.triton/nvidia

COPY code .

RUN ln -sf /usr/lib/aarch64-linux-gnu/libstdc++.so.6 $CONDA_DIR/envs/dlcompiler/lib/libstdc++.so.6 && \
    ldconfig

RUN set -ex && \
    source ~/.bashrc && \
    conda activate dlcompiler && \
    cd /root/workspace && \
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH && \
    ln -sf /usr/lib/aarch64-linux-gnu/libstdc++.so.6 $CONDA_DIR/envs/dlcompiler/lib/libstdc++.so.6 && \
    ldconfig && \
    echo "当前LD_LIBRARY_PATH: $LD_LIBRARY_PATH" && \
    echo "Python路径: $(which python)" && \
    echo "PIP路径: $(which pip)" && \
    echo y | bash compile.sh
