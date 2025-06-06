#!/bin/bash
set -ex

export LLVM_BUILD_DIR=/triton_depends/llvm-20-install

if [ -d ${WORKSPACE}triton ];then
  rm -rf ${WORKSPACE}triton
fi

if [ -d ~/.triton/dump ];then
  rm -rf ~/.triton/dump
fi

if [ -d ~/.triton/cache ];then
  rm -rf ~/.triton/cache
fi

cd ${WORKSPACE}
git clone --depth 1 https://gitee.com/shijingchang/triton.git
cd triton/python
TRITON_PLUGIN_DIRS=${WORKSPACE}/ascend \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
pip install -e . --no-build-isolation -v

if [ -d __pycache__ ];then
  rm -rf __pycache__
fi
if [ -d ${HOME}/.triton/cache ]; then
  rm -rf ${HOME}/.triton/cache
fi

pip list

cd ${WORKSPACE}

# 定义日志文件名（格式示例：test_results_20231015_153045.log）
LOG_FILE="${WORKSPACE}/test_results_$(date +%Y%m%d).log"

# 定义要运行的测试目录
TEST_generalization="${WORKSPACE}/ascend/examples/generalization_cases"
TEST_prof="${WORKSPACE}/ascend/examples/prof_cases"
pwd
ls -al
# 记录测试开始时间
echo "===== 测试开始时间: $(date +"%Y-%m-%d %H:%M:%S") =====" > "$LOG_FILE"

#函数:运行测试并返回状态码
function run_tests() {
  local dir="$1"
  echo "Running tests in dir: $dir" | tee -a "$LOG_FILE"
  cd ${dir} || {
    echo "failed to cd to : ${dir}" | tee -a "$LOG_FILE"
    return 1
    }

  #运行测试
  pytest -sv -n 16 . 2>&1 | tee -a "$LOG_FILE"

  local pytest_exit=$?
  return $pytest_exit
}
cd ${WORKSPACE}
tree
run_tests "${TEST_generalization}"
r1=$?
cd ${WORKSPACE}
ls -al
run_tests "${TEST_prof}"
r2=$?
echo -e "\n===== 测试结束时间: $(date +"%Y-%m-%d %H:%M:%S") =====" >> "$LOG_FILE"
if [ $r1 -ne 0 ] || [ $r2 -ne 0 ]; then
  echo "some tests failed. Check log for details."
  cp "$LOG_FILE" "/triton_depends/daily_log"
  exit 1
else
  echo "all tests passed."
  cp "$LOG_FILE" "/triton_depends/daily_log"
fi

