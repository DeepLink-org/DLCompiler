# export LLVM_BINARY_DIR=/mnt/data01/zmz/workspace/04ttshared/llvm-install/bin
export TRITON_PLUGIN_DIRS=$(pwd)
export GOOGLETEST_DIR=/mnt/data01/zmz/workspace/04ttshared/test/googletest
pip uninstall triton -y
# cd third_party/triton_shared/triton/python/
cd third_party/triton/python/
rm -rf build/
# cd /mnt/data01/zmz/workspace/04ttshared/Triton/third_party/triton_shared/triton/python/triton/backends 
# unlink dicp_triton
# ln -s /mnt/data01/zmz/workspace/04ttshared/Triton/backend dicp_triton
# cd -
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install --no-build-isolation -vvv .[tests] -i https://mirrors.huaweicloud.com/repository/pypi/simple
