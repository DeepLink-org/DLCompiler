# Triton
triton for dsa

## Usage

## 安装 clang
```
apt-get update
apt-get install clang
apt-get install python3.10
```

## 下载 triton
```
git clone git@github.com:triton-lang/triton.git
cat triton/cmake/llvm-hash.txt
```


## compile llvm project
```
git clone https://github.com/llvm/llvm-project.git
// triton下的llvm-hash.txt commit id
git reset --hard 6f44bb7717897191be25aa01161831c67cdf5b84

cmake -G Ninja ../llvm  -DLLVM_ENABLE_PROJECTS="llvm;mlir"    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"    -DMLIR_ENABLE_CUDA_RUNNER=ON    -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_ASSERTIONS=ON    -DCMAKE_C_COMPILER=clang    -DCMAKE_CXX_COMPILER=clang++    -DLLVM_INSTALL_UTILS=ON

ninja -j64
```


## 编译 pybind
```
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake ..
make check -j16
make install
```

## 编译 triton && triton
```
cd dicp_triton/third_party/triton
mkdir build
// cmake 命令
/home/yuansheng/software/cmake-3.28.0-linux-x86_64/bin/cmake .. -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_ENABLE_WERROR=ON -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/yuansheng/workspace/dicp/dicp_triton/_C -DTRITON_BUILD_TUTORIALS=OFF -DTRITON_BUILD_PYTHON_MODULE=ON -DPython3_EXECUTABLE:FILEPATH=/usr/bin/python3 -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DPYTHON_INCLUDE_DIRS=/usr/local/include/python3.10 -DTRITON_CODEGEN_BACKENDS="nvidia;amd" -DTRITON_PLUGIN_DIRS=/home/yuansheng/workspace/dicp/dicp_triton  -DLLVM_INCLUDE_DIRS=/nvme/nvme2/share/share_data/yuansheng/llvm-project/build/include -DLLVM_LIBRARY_DIR=/nvme/nvme2/share/share_data/yuansheng/llvm-project/build/lib -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts -DMLIR_DIR=/nvme/nvme2/share/share_data/yuansheng/llvm-project/build/lib/cmake/mlir -DCUPTI_INCLUDE_DIR=/usr/local/cuda/include -DJSON_INCLUDE_DIR=/home/yuansheng/workspace/dicp/dicp_triton/third_party/json/include


ninja -j64
```

## 设置环境变量
```
export PYTHONPATH=/home/yuansheng/workspace/dicp/dicp_triton/third_party/triton/python:/home/yuansheng/workspace/dicp/dicp_triton
```