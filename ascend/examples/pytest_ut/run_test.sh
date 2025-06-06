#!/bin/bash

set -ex

function uninstall_triton_ascend() {
  set +e
  while true; do
    pip3 uninstall triton_ascend -y | grep "Found existing installation"
    if [ $? -eq 1 ]; then
        echo "All triton_ascend versions are uninstalled"
        break
    fi
  done
  set -e
}

function build_and_test() {
  if [ -d ${HOME}/.triton/dump ];then
    rm -rf ${HOME}/.triton/dump
  fi
  if [ -d ${HOME}/.triton/cache ];then
    rm -rf ${HOME}/.triton/cache
  fi

  cd ${WORKSPACE}
  # Run uninstall once because the while-loop does not stop. No idea why.
  # uninstall_triton_ascend
  pip3 uninstall triton_ascend -y

  git submodule set-url third_party/triton https://gitee.com/shijingchang/triton.git
  git submodule sync && git submodule update --init --recursive

  bash scripts/build.sh ${WORKSPACE}/ascend ${LLVM_BUILD_DIR} 3.2.0 install 1

  cd ${WORKSPACE}/ascend/examples/pytest_ut
  pytest -n 16 --dist=load . || { exit 1 ; }
}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LLVM_BUILD_DIR=/opt/llvm-b5cc222

# FIXME: 20250508 the bishengir-compile in the CANN 8.0.T115 fails lots of cases
#        So we need to use another version of compiler.
COMPILER_ROOT=/home/shared/bisheng_toolkit_20250519
export PATH=${COMPILER_ROOT}:${COMPILER_ROOT}/ccec_compiler/bin:$PATH

# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_and_test
