#!/usr/bin/env bash

# ===== 环境变量配置 =====

export JSON_PATH34=/root/workspace/v34/include.zip
export GOOGLETEST_DIR34=/root/workspace/v34/googletest
export LLVM_TGZ_PATH34=/root/workspace/v34/llvm-064f02da-ubuntu-arm64.tar.gz
export DEBUG=1
export IS_NOT_PUBLISH=0
# bash compile_shared.sh apply_patch=true compile_triton_shared=true
rm -rf ~/.triton/cache/

# ===== 使用 expect 自动回应 =====
expect <<'EXPECT_SCRIPT'
  log_user 1
  set timeout -1   ;# 不限制时间，直到编译结束
  spawn bash compile_shared.sh 
  # spawn bash compile_shared.sh

  expect {
    "即将清空third_party下面的源码改动然后apply patch, 是否继续? (y/n)" {
      send "y\r"
      exp_continue
    }
    eof
  }
EXPECT_SCRIPT

python test/dsl/hopper-fa-ws-pipelined-pingpong_test.py