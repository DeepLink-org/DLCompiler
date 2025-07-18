# export GOOGLETEST_DIR=/path/to/googletest
export TRITON_PLUGIN_DIRS=$(pwd)
pip uninstall triton -y
cd third_party/triton_shared/
git apply $TRITON_PLUGIN_DIRS/patch/triton_shared_changes.patch
git apply $TRITON_PLUGIN_DIRS/patch/triton_changes.patch
cd $TRITON_PLUGIN_DIRS/third_party/triton/python/
rm -rf build/
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install --no-build-isolation -vvv .[tests] -i https://mirrors.huaweicloud.com/repository/pypi/simple
