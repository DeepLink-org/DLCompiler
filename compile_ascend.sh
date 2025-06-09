export LLVM_INSTALL_PREFIX=/mnt/data01/zmz/software/llvm-install

# cmake ../llvm \
#   -G Ninja \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DLLVM_ENABLE_ASSERTIONS=ON \
#   -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
#   -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
#   -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX} \
#   -DCMAKE_C_COMPILER=clang \
#   -DCMAKE_CXX_COMPILER=clang++
# ninja install

cd third_party/triton-ascend/
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_PLUGIN_DIRS=./ascend \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install

cd ../..
python -c "import triton; print(triton.__path__)" 
