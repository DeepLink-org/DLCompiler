#!/bin/bash
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"

# SET ENV
# export JSON_PATH=/path/to/your/json/file
# export GOOGLETEST_DIR=/path/to/your/googletest/directory
# export LLVM_BUILD_DIR=/path/to/your/llvm-project/build
# export LLVM_TGZ_PATH=/path/to/your/llvm-86b69c31-ubuntu-arm64.tar.gz      # 可选，用于指定LLVM的tgz包路径
export TRITON_PLUGIN_DIRS=$(pwd)
apply_patch=false

# 解析命令行参数
if [ "$1" = "apply_patch=true" ]; then
    apply_patch=true
fi

if [[ $apply_patch == true ]]; then
    # 交互式询问是否可以修改triton/triton_shared目录下代码
    read -p "即将清空third_party下面的源码改动然后apply patch, 是否继续? (y/n)" -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # do dangerous stuff
        echo "Apply triton and triton_shared patch"
        cd $TRITON_PLUGIN_DIRS/third_party/triton_shared/
        git checkout .
        ls $TRITON_PLUGIN_DIRS/patch/ttshared/*.patch | xargs -n1 git apply
        if [ $? -ne 0 ]; then
            echo "Error: triton_shared git apply failed." >&2
            exit 1
        fi
        cd $TRITON_PLUGIN_DIRS/third_party/triton/
        git checkout .
        ls $TRITON_PLUGIN_DIRS/patch/triton/*.patch | xargs -n1 git apply
        if [ $? -ne 0 ]; then
            echo "Error: triton git apply failed." >&2
            exit 1
        fi
    else
        # do dangerous stuff
        echo "没有apply patch/*.patch, 已退出!"
        exit 1
    fi
fi

pip uninstall triton -y

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
if [[ $apply_patch == true ]]; then
    echo "编译前先清空了third_party源码改动, 然后执行了apply patch/*.patch, 请检查正确性!"
else
    echo "编译前没有执行apply patch/*.patch, 请检查正确性!"
fi