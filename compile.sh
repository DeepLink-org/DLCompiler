export DICP_TRITON_DIR=$PWD
pip3 install pybind11 ninja
git submodule update --init

cd $DICP_TRITON_DIR/third_party/triton

rm -rf build/third_party/dicp_triton/
mkdir -p build && cd build

cmake .. -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_ENABLE_WERROR=ON \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$DICP_TRITON_DIR/third_party/triton/python/triton/_C \
    -DTRITON_BUILD_TUTORIALS=OFF -DTRITON_BUILD_PYTHON_MODULE=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
    -DTRITON_PLUGIN_DIRS=$DICP_TRITON_DIR -DLLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
    -DLLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib  -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DTRITON_BUILD_PROTON=0 -DJSON_INCLUDE_DIR=$DICP_TRITON_DIR/third_party/json/include -DTRITON_BUILD_UT=0

ninja -j32

cd ../python/triton/backends/
unlink dicp_triton
ln -s $DICP_TRITON_DIR/backend dicp_triton