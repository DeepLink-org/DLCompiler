# Triton
triton for dsa

## compile llvm project
```
git clone https://github.com/llvm/llvm-project.git
// triton下的llvm-hash.txt commit id
git reset --hard ed4e505c219fe6c7464ea5a056e90d8cd94c7332

cmake -G Ninja ../llvm  -DLLVM_ENABLE_PROJECTS="llvm;mlir"    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86X86;NVPTX;AMDGPU"     -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_ASSERTIONS=ON       -DLLVM_INSTALL_UTILS=ON

ninja -j64
```


## 编译 triton && triton
```
export LLVM_BUILD_DIR=...
bash compile.sh
export PYTHONPATH=$PWD/third_party/triton/python
export PATH=$PWD/third_party/triton/build/third_party/triton_shared/tools/triton-shared-opt/:$PATH
```


## 测试
```
cd python/op
python softmax.py
```