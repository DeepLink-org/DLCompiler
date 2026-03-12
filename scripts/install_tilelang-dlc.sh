#!/bin/bash

# Add command line option parsing
USE_LLVM=false
DEBUG_MODE=false
REINSTALL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug|-d)
            DEBUG_MODE=true
            shift
            ;;
        --reinstall|-r)
            REINSTALL=true
            shift
            ;;
        --enable-llvm)
            USE_LLVM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--enable-llvm]"
            exit 1
            ;;
    esac
done

echo "Starting installation script..."

if [ -z "${TILELANG_DLC_PATH}" ]; then
    echo "Error: 环境变量 TILELANG_DLC_PATH 未设置" >&2
    exit 1
fi

if [ -z "${DLCOMPILER_SOURCE}" ]; then
    echo "Error: 环境变量 DLCOMPILER_SOURCE 未设置" >&2
    exit 1
fi

cd "${TILELANG_DLC_PATH}" || {
    echo "Error: Can not cd to dir: '${TILELANG_DLC_PATH}'" >&2
    exit 1
}

echo "pwd: $(pwd)"

# Step 1: Install Python requirements
echo "Installing Python requirements from requirements.txt..."
pip install -r requirements-dev.txt -i  https://mirrors.huaweicloud.com/repository/pypi/simple
pip install -r requirements.txt -i  https://mirrors.huaweicloud.com/repository/pypi/simple
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python requirements."
    exit 1
else
    echo "Python requirements installed successfully."
fi


# # Step 9: Clone and build TVM
# echo "Cloning TVM repository and initializing submodules..."
# # clone and build tvm
# git submodule update --init --recursive

if [ "$REINSTALL" != "true" ]; then
    if [ -d build ]; then
        rm -rf build
    fi
fi

mkdir -p build
# cp 3rdparty/tvm/cmake/config.cmake build
cd build

# echo "set(USE_COMMONIR ON)" >> config.cmake

# Define common CMake parameters as array
LLVM_CMAKE_PARAMS=(
    "-DLLVM_BUILD_EXAMPLES=ON"
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DLLVM_INSTALL_UTILS=ON"
)

echo "Running CMake for TileLang..."
declare -a cmake_params
if $DEBUG_MODE; then
    # Debug mode with additional flags
    cmake_params+=("-DCMAKE_BUILD_TYPE=Debug")
    cmake_params+=("-DCMAKE_CXX_FLAGS=-g3 -fno-omit-frame-pointer")
    cmake_params+=("-DCMAKE_C_FLAGS=-g3 -fno-omit-frame-pointer -fno-optimize-sibling-calls")
else
    # Release mode
    :
fi

# Run CMake with proper array expansion
cmake .. "${cmake_params[@]}"

if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

echo "Building TileLang with make..."

# Calculate 75% of available CPU cores
# Other wise, make will use all available cores
# and it may cause the system to be unresponsive
CORES=$(nproc)
MAKE_JOBS=$(( CORES * 75 / 100 ))
make -j${MAKE_JOBS}

if [ $? -ne 0 ]; then
    echo "Error: TileLang build failed."
    exit 1
else
    echo "TileLang build completed successfully."
fi

cd ..

# Step 11: Set environment variables
TILELANG_PATH="$(pwd)"
if [ "$REINSTALL" != "true" ]; then
    if ! grep -q "# TileLang PYTHONPATH" ~/.bashrc; then
        echo "Configuring environment variables for TVM..."
        echo "export PYTHONPATH=${TILELANG_PATH}:\$PYTHONPATH # TileLang PYTHONPATH" >> ~/.bashrc
        echo "TileLang environment variables added to ~/.bashrc"
    else
        echo "TileLang environment variables already configured in ~/.bashrc"
    fi
fi

# Step 12: Source .bashrc to apply changes
echo "Applying environment changes by sourcing .bashrc..."
source ~/.bashrc
if [ $? -ne 0 ]; then
    echo "Error: Failed to source .bashrc."
    exit 1
else
    echo "Environment configured successfully."
fi

echo "Installation script completed successfully."
