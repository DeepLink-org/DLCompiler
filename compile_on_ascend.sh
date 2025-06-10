
export LLVM_INSTALL_PREFIX=/mnt/data01/zmz/software/llvm-install

change_ascend() {
    file_path="third_party/triton-ascend/ascend/triton-adapter/include/TritonToLinalg/Passes.h"
    old_line="#include \"ascend/triton-adapter/include/TritonToLinalg/Passes.h.inc\""
    new_line="#include \"dicp_triton/triton-adapter/include/TritonToLinalg/Passes.h.inc\""

    if [ -f "$file_path" ]; then
        sed -i "s|$old_line|$new_line|g" "$file_path"
        echo "file $file_path change successfully."
    else
        echo "can not find $file_path, need check."
    fi
}

# 1. compile llvm
# cd path/to/llvm
# mkdir build && cd build
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

# 2. use dicp_triton/backend change triton backend
cp backend/* third_party/triton-ascend/ascend/backend/
cp dicp_triton.cc third_party/triton-ascend/ascend/triton_ascend.cpp
cp setup.py third_party/triton-ascend/
change_ascend

# 3. compile triton
cd third_party/triton-ascend/
rm -rf build

LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_PLUGIN_DIRS=./ascend \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install

# 4. check triton
cd ../../python/op
python -c "import triton; print(triton.__path__)" 
python softmax.py
echo "softmax.py test done"
