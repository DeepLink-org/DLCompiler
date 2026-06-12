#!/bin/bash
set -e

# Conda 安装目录（可自定义）
CONDA_DIR="${CONDA_DIR:-/opt/conda}"

echo "==> 1. 下载并安装 Miniconda"
if [ -d "$CONDA_DIR" ]; then
    echo "    已存在 $CONDA_DIR，跳过安装"
else
    ARCH=$(uname -m)
    MINI_ARCH="$ARCH"
    [ "$ARCH" = "aarch64" ] && MINI_ARCH="aarch64"
    [ "$ARCH" = "x86_64" ]  && MINI_ARCH="x86_64"

    wget -q "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-${MINI_ARCH}.sh" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm -f /tmp/miniconda.sh
fi

export PATH="$CONDA_DIR/bin:$PATH"

echo "==> 2. 配置 Conda 使用清华源"
# 系统级 .condarc 中的 defaults 会导致访问 repo.anaconda.com 并要求接受 ToS，
# 直接覆盖系统级与用户级 .condarc，只保留清华镜像
cat > "$CONDA_DIR/.condarc" <<'EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
show_channel_urls: true
EOF
cat > "$HOME/.condarc" <<'EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
show_channel_urls: true
EOF

echo "==> 3. 创建 dlcompiler 环境 (Python 3.10)"
if conda env list | awk '{print $1}' | grep -qx "dlcompiler"; then
    echo "    已存在 dlcompiler 环境，跳过创建"
else
    conda create -n dlcompiler python=3.10 -y
fi

# 激活环境
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate dlcompiler

echo "==> 4. 配置 pip 使用清华源"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

echo "==> 5. 安装 Triton 构建依赖"
pip install --no-cache-dir \
    autopep8 isort numpy pytest pytest-forked pytest-xdist \
    "scipy>=1.7.1" llnl-hatchet expecttest \
    "setuptools>=40.8.0" wheel "cmake>=3.20,<4.0" \
    "ninja>=1.11.1" "pybind11>=2.13.1" lit nanobind

echo "==> 6. 安装 PyTorch (CPU 版) 与 torch_npu (通过 pip 指定版本，不下载 wheel)"
# 安装 PyTorch CPU 版本，使用官方索引
pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 安装 torch_npu，默认从 PyPI 获取（若需 Ascend 专用源可加 --extra-index-url）
pip install torch-npu==2.9.0.post2

echo "==> 7. 安装额外 DLCompiler 依赖 (requirements.txt)"
if [ -f requirements.txt ]; then
    pip install --no-cache-dir -r requirements.txt
else
    echo "警告: 未找到 requirements.txt 文件，跳过该步骤"
fi

echo "==> 环境安装完成！请执行以下命令激活 dlcompiler 环境："
echo "    source $CONDA_DIR/etc/profile.d/conda.sh"
echo "    conda activate dlcompiler"