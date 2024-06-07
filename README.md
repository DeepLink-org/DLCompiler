# Triton
triton for dsa

## Usage

# compile llvm project

// 安装 clang
apt-get update
apt-get install clang

// clone llvm project
git clone https://github.com/llvm/llvm-project.git

// 查看triton中
// 查看 /home/sheng.yuan/workspace/triton/cmake/llvm-hash.txt
// 回退llvm project的commit id
cd llvm_project
git reset --hard 6f44bb7717897191be25aa01161831c67cdf5b84

// cmake 命令
cmake -G Ninja ../llvm  -DLLVM_ENABLE_PROJECTS="llvm;mlir"    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"    -DMLIR_ENABLE_CUDA_RUNNER=ON    -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_ASSERTIONS=ON    -DCMAKE_C_COMPILER=clang    -DCMAKE_CXX_COMPILER=clang++    -DLLVM_INSTALL_UTILS=ON

// 编译 llvm project
ninja -j64

# 编译 pybind
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake ..
make check -j16
make install

# 编译 pybind
// 修改CMakeLists.txt
set(PYBIND11_INCLUDE_DIR /home/sheng.yuan/workspace/pybind11/include)
set(MLIR_DIR /home/sheng.yuan/workspace/llvm-project/build/lib/cmake/mlir)

// cmake 命令
cmake .. -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_ENABLE_WERROR=ON -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/sheng.yuan/workspace/triton/python/triton/_C -DTRITON_BUILD_TUTORIALS=OFF -DTRITON_BUILD_PYTHON_MODULE=ON -DPython3_EXECUTABLE:FILEPATH=/usr/bin/python3 -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DPYTHON_INCLUDE_DIRS=/usr/include/python3.10 -DTRITON_CODEGEN_BACKENDS="nvidia;amd" -DTRITON_PLUGIN_DIRS=/home/sheng.yuan/workspace/Triton -DLLVM_INCLUDE_DIRS=/home/sheng.yuan/workspace/llvm-project/build/include -DLLVM_LIBRARY_DIR=/home/sheng.yuan/workspace/llvm-project/build/lib -DPYBIND11_INCLUDE_DIR=/home/sheng.yuan/workspace/pybind11/include -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts -DMLIR_DIR=/home/sheng.yuan/workspace/llvm-project/build/lib/cmake/mlir

// 如果需要编译triton plugin 

// 编译 llvm project
ninja -j64