#!/bin/bash
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"

home_path=$(pwd)
# compile triton shared library with patch
compile_triton_shared=false
apply_patch=false

# 遍历所有参数
for arg in "$@"; do
    if [ "$arg" = "compile_triton_shared=true" ]; then
        echo "检测到 compile_triton_shared=true"
        compile_triton_shared=true
    elif [ "$arg" = "apply_patch=true" ]; then
        echo "检测到 apply_patch=true"
        apply_patch=true
    fi
done


echo "start compile ========================================"
echo compile_triton_shared: $compile_triton_shared
echo apply_patch: $apply_patch
echo "======================================================"

if [[ $compile_triton_shared == true ]]; then
    echo "start compile triton_shared"
    cd third_party
    mkdir -p build && rm -rf build/*
    git clone --no-hardlinks triton_shared build/triton_shared
    git clone --no-hardlinks triton build/triton
    cd build
    export TRITON_PLUGIN_DIRS=$(pwd)/triton_shared
    cd triton_shared && git clean -xdf && git checkout . && git checkout 2b728ad97bc02af821a0805b09075838911d4c19 && ls ../../../patch/v3_4/triton_shared.patch | xargs -n1 git apply && cd ../
    cd triton && git clean -xdf && git checkout . && cd ../
    cd triton && git checkout $(cat ../triton_shared/triton-hash.txt) && ls ../../../patch/v3_4/triton.patch | xargs -n1 git apply
    TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install --no-build-isolation -vvv '.[tests]'
    if [ $? -ne 0 ]; then
        echo "Error: triton_shared compile failed." >&2
        exit $?
    fi
    echo "triton_shared compile success!"
fi

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

pip uninstall triton -y

cd $TRITON_PLUGIN_DIRS/third_party/triton/
rm -rf build/

if [ -z "$LLVM_BUILD_DIR" ]; then
    TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true \
    python3 -m pip install --no-build-isolation -vvv .[tests] -i https://mirrors.huaweicloud.com/repository/pypi/simple
    if [ $? -ne 0 ]; then
        notify_apply_patch
        echo "Error: DLCompiler compile failed." >&2
        exit $?
    fi
    # echo "LLVM_BUILD_DIR is not set, using system LLVM or downloading prebuilt LLVM."
    # TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true \
    # python3 -m pip wheel --no-deps --no-build-isolation -w dist/ .
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
        echo "Error: DLCompiler compile failed." >&2
        exit $?
    fi
fi
notify_apply_patch