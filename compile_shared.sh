#!/bin/bash
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"

# SET ENV
# export JSON_PATH=/path/to/your/json/file
# export GOOGLETEST_DIR=/path/to/your/googletest/directory
# export LLVM_BUILD_DIR=/path/to/your/llvm-project/build
export TRITON_PLUGIN_DIRS=$(pwd)

# 交互式询问是否可以修改triton/triton_shared目录下代码
read -p "是否可以修改triton/triton_shared目录下代码？(y/n)" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # do dangerous stuff
    echo "应用triton/triton_shared patch"
else
    # do dangerous stuff
    echo "请先备份triton/triton_shared目录下代码，然后再执行此脚本"
    exit 1
fi
pip uninstall triton -y
cd third_party/triton_shared/


# apply patch
git checkout .
git apply $TRITON_PLUGIN_DIRS/patch/ttshared_changes.patch
if [ $? -ne 0 ]; then
    echo "Error: triton_shared git apply failed." >&2
    exit 1
fi

cd $TRITON_PLUGIN_DIRS/third_party/triton/
git checkout .
# git apply $TRITON_PLUGIN_DIRS/patch/triton_changes.patch
ls $TRITON_PLUGIN_DIRS/patch/triton_*.patch | xargs -n1 git apply
if [ $? -ne 0 ]; then
    echo "Error: triton git apply failed." >&2
    exit 1
fi
cd $TRITON_PLUGIN_DIRS/third_party/triton/python/
rm -rf build/

if [ -z "$LLVM_BUILD_DIR" ]; then
    TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true \
    python3 -m pip install --no-build-isolation -vvv .[tests] -i https://mirrors.huaweicloud.com/repository/pypi/simple
else
    echo "LLVM_BUILD_DIR is set to $LLVM_BUILD_DIR"
    LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
    LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
    LLVM_SYSPATH=$LLVM_BUILD_DIR \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_WITH_CCACHE=true \
    python3 -m pip install --no-build-isolation -vvv .[tests] -i https://mirrors.huaweicloud.com/repository/pypi/simple
fi
