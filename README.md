# Triton
triton for dsa

## Usage

## 安装 clang
```
apt-get update
apt-get install clang
```

## compile llvm project
```
git clone https://github.com/llvm/llvm-project.git
// triton下的llvm-hash.txt commit id
git reset --hard 6f44bb7717897191be25aa01161831c67cdf5b84

cmake -G Ninja ../llvm  -DLLVM_ENABLE_PROJECTS="llvm;mlir"    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"    -DMLIR_ENABLE_CUDA_RUNNER=ON    -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_ASSERTIONS=ON    -DCMAKE_C_COMPILER=clang    -DCMAKE_CXX_COMPILER=clang++    -DLLVM_INSTALL_UTILS=ON

ninja -j64
```


## 编译 triton && triton
```
export LLVM_BUILD_DIR=...
export CUPTI_INCLUDE_DIR=...
bash compile.sh
export PYTHONPATH=$PWD/third_party/triton/python
export PATH=$PWD/third_party/triton/build/third_party/triton_linalg/bin:$PATH
```


## 测试
```
cd python/op
python softmax.py
```