ARG TARGETPLATFORM=linux/arm64
FROM --platform=${TARGETPLATFORM} swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11-devel

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ============================================================
# 1. 替换 Ubuntu 源为阿里源
# ============================================================
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    cat > /etc/apt/sources.list << 'EOF'
deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
EOF
RUN apt-get update

# ============================================================
# 2. 安装基础依赖
# ============================================================
RUN apt-get install -y --no-install-recommends \
    wget curl git build-essential software-properties-common \
    apt-transport-https ca-certificates gnupg lsb-release cmake
# ============================================================
# 5. 安装其他构建依赖
# ============================================================
RUN apt-get install -y --no-install-recommends \
    ninja-build  python3-pip libssl-dev libffi-dev \
    bash make sudo unzip vim zlib1g-dev
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================================
# 3. 安装 LLVM/Clang (>=15)，固定为 15 版本
#    对应原脚本步骤3，但只装15且使用阿里云源
# ============================================================
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    cat > /etc/apt/sources.list.d/llvm-aliyun.list << 'EOF'
deb http://mirrors.aliyun.com/llvm/apt/jammy/ llvm-toolchain-jammy-15 main
EOF
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        clang-15 clangd-15 lld-15 \
        clang-format-15 clang-tidy-15 \
        llvm-15-dev libclang-15-dev
RUN update-alternatives --install /usr/bin/clang        clang        /usr/bin/clang-15        100 && \
    update-alternatives --install /usr/bin/clang++      clang++      /usr/bin/clang++-15      100 && \
    update-alternatives --install /usr/bin/clangd       clangd       /usr/bin/clangd-15       100 && \
    update-alternatives --install /usr/bin/lld          lld          /usr/bin/lld-15          100 && \
    update-alternatives --install /usr/bin/ld.lld       ld.lld       /usr/bin/ld.lld-15       100 && \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-15 100 && \
    update-alternatives --install /usr/bin/clang-tidy   clang-tidy   /usr/bin/clang-tidy-15   100




# ============================================================
# 6. 安装 Miniconda
# ============================================================
ENV CONDA_DIR=/opt/conda
RUN ARCH=$(uname -m) && \
    MINI_ARCH=$ARCH && \
    [ "$ARCH" = "aarch64" ] && MINI_ARCH="aarch64" && \
    [ "$ARCH" = "x86_64" ]  && MINI_ARCH="x86_64" && \
    wget -q "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-${MINI_ARCH}.sh" -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -f /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -tipy && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate dlcompiler" >> ~/.bashrc
ENV PATH="$CONDA_DIR/bin:$PATH"
# 直接覆盖 .condarc，避免保留默认的 defaults channel
# （新版 conda 访问 repo.anaconda.com 时会要求接受 ToS，导致 conda create 失败）
RUN cat > $CONDA_DIR/.condarc <<'EOF' && cp $CONDA_DIR/.condarc /root/.condarc
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
show_channel_urls: true
EOF

# ============================================================
# 7. 创建并激活 dlcompiler 环境（python=3.10），pip 改为清华源
# ============================================================
RUN conda create -n dlcompiler python=3.10 -y

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate dlcompiler && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# ============================================================
# 8. 安装 Triton 构建依赖
# ============================================================
RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate dlcompiler && \
    pip install --no-cache-dir \
        autopep8 isort numpy pytest pytest-forked pytest-xdist \
        "scipy>=1.7.1" llnl-hatchet expecttest \
        "setuptools>=40.8.0" wheel "cmake>=3.20,<4.0" \
        "ninja>=1.11.1" "pybind11>=2.13.1" lit nanobind

# ============================================================
# 9. 安装 PyTorch + torch_npu
# ============================================================
RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate dlcompiler && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        wget -q -P /tmp https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl && \
        pip install /tmp/torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl && \
        rm -f /tmp/torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl && \
        wget -q -P /tmp https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.9.0/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl && \
        pip install /tmp/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl && \
        rm -f /tmp/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl; \
    elif [ "$ARCH" = "x86_64" ]; then \
        wget -q -P /tmp https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl && \
        pip install /tmp/torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl && \
        rm -f /tmp/torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl && \
        wget -q -P /tmp https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.9.0/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_x86_64.whl && \
        pip install /tmp/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_x86_64.whl && \
        rm -f /tmp/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_x86_64.whl; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi

# ============================================================
# 10. 设置工作目录和 CANN 环境
# ============================================================
WORKDIR /root/workspace
RUN echo "alias ll='ls -alh --color=auto'" >> ~/.bashrc && \
    echo "alias ls='ls --color=auto'" >> ~/.bashrc && \
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:\$LD_LIBRARY_PATH" >> ~/.bashrc

# 额外 DLCompiler 依赖（原 Dockerfile 保留）
COPY requirements.txt .
RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate dlcompiler && \
    pip install --no-cache-dir -r requirements.txt
