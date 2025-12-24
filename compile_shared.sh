#!/bin/bash
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"

home_path=$(pwd)
# compile triton shared library with patch
apply_patch=false
build_package=false

# 遍历所有参数
for arg in "$@"; do
    if [ "$arg" = "apply_patch=true" ]; then
        echo "检测到 apply_patch=true"
        apply_patch=true
    elif [ "$arg" = "build_package=true" ]; then
        echo "检测到 build_package=true"
        build_package=true
    fi
done


echo "start compile ========================================"
echo apply_patch: $apply_patch
echo "======================================================"

echo "start apply ascendnpu-ir patch"
cd $home_path/third_party/ascendnpu-ir
git checkout .
git apply ../../patch/ascendnpu-ir.patch
echo "apply patch/ascendnpu-ir.patch success!"

# SET ENV
# export JSON_PATH=/path/to/your/json/file
# export GOOGLETEST_DIR=/path/to/your/googletest/directory
# export LLVM_BUILD_DIR=/path/to/your/llvm-project/build
# export LLVM_TGZ_PATH=/path/to/your/llvm-86b69c31-ubuntu-arm64.tar.gz      # 可选，用于指定LLVM的tgz包路径
export TRITON_PLUGIN_DIRS=$home_path

is_npu=false
check_npu() {
    if command -v npu-smi info &> /dev/null && npu-smi info &> /dev/null; then
        is_npu=true
    else
        is_npu=false
    fi
}

check_npu

if [[ $apply_patch == true ]]; then
    # do dangerous stuff
    echo "Apply triton and triton_shared patch"
    echo "当前环境检测为：$([[ $is_npu == true ]] && echo 'ascend加速卡，使用适配patch' || echo '非ascend加速卡，不使用适配patch')"
    if [[ $is_npu == true ]]; then
        cd $TRITON_PLUGIN_DIRS/third_party/triton_shared/
        git checkout .
        ls $TRITON_PLUGIN_DIRS/patch/ttshared/*.patch | xargs -n1 git apply
        if [ $? -ne 0 ]; then
            echo "Error: triton_shared git apply failed." >&2
            exit 1
        fi
    fi
    cd $TRITON_PLUGIN_DIRS/third_party/triton/
    git checkout .
    ls $TRITON_PLUGIN_DIRS/patch/triton/*.patch | xargs -n1 git apply
    if [ $? -ne 0 ]; then
        echo "Error: triton git apply failed." >&2
        exit 1
    fi
fi

notify_apply_patch() {
    if [[ $apply_patch == true ]]; then
        echo "编译前先清空了third_party源码改动, 然后执行了apply patch/*.patch, 请检查正确性!"
    else
        echo "编译前没有执行apply patch/*.patch, 请检查正确性!"
    fi
}

if [[ $build_package == true ]]; then
    echo "build_package is true, skip uninstall triton dlcompiler"
    cd $TRITON_PLUGIN_DIRS/third_party/triton/
else
    pip uninstall triton dlcompiler -y

    cd $TRITON_PLUGIN_DIRS/third_party/triton/
    rm -rf build/
fi

if [ -z "$LLVM_BUILD_DIR" ]; then
    # 使用build_package参数，控制是否进行编译
    if [[ $build_package == true ]]; then
        echo "LLVM_BUILD_DIR is not set, using system LLVM or downloading prebuilt LLVM build package."
        TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true \
        python3 -m pip wheel --no-deps --no-build-isolation -w dist/ .
    else
        TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true \
        python3 -m pip install --no-build-isolation -vvv .[tests] -i https://mirrors.huaweicloud.com/repository/pypi/simple
        if [ $? -ne 0 ]; then
            notify_apply_patch
            echo "Error: DLCompiler compile failed."
            exit 1
        fi
    fi
else
    echo "LLVM_BUILD_DIR is set to $LLVM_BUILD_DIR"
    LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
    LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
    LLVM_SYSPATH=$LLVM_BUILD_DIR \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_WITH_CCACHE=true \
    python3 -m pip install --no-build-isolation -vvv .[tests] -i https://mirrors.huaweicloud.com/repository/pypi/simple
    if [ $? -ne 0 ]; then
        notify_apply_patch
        echo "Error: DLCompiler compile failed."
        exit 1
    fi
fi
notify_apply_patch
