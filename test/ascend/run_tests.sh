#!/bin/bash

set -euo pipefail
script=$(readlink -f "$0")
script_dir=$(dirname "$script")

function run_pytestcases() {
  PRIMARY_ASCEND_DEVICE="${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
  FALLBACK_ASCEND_DEVICE="${ASCEND_FALLBACK_RT_VISIBLE_DEVICES:-5}"

  if [ -d ${HOME}/.triton/dump ]; then
    rm -rf ${HOME}/.triton/dump
  fi
  if [ -d ${HOME}/.triton/cache ]; then
    rm -rf ${HOME}/.triton/cache
  fi

  cd ${script_dir}
  TARGET_DIR="$1"
  cd ${TARGET_DIR}

  echo "[Phase 1] Run tests in parallel on Ascend device ${PRIMARY_ASCEND_DEVICE}"
  set +e
  timeout --signal=TERM 20m  pytest . -n 8 --dist=loadscope --reruns 1 --reruns-delay 2
  parallel_rc=$?
  set -e

  if [ "${parallel_rc}" -eq 0 ]; then
    echo "[SUCCESS] All tests passed"
    return 0
  fi
  if [ "${parallel_rc}" -eq 124 ]; then
    echo "[ERROR] Parallel run timed out"
    return "${parallel_rc}"
  fi

  echo "[INFO] Failed cases collected from device ${PRIMARY_ASCEND_DEVICE}:"
  if [ -f .pytest_cache/v/cache/lastfailed ]; then
    sed -n '1,200p' .pytest_cache/v/cache/lastfailed
  else
    echo "[WARN] .pytest_cache/v/cache/lastfailed not found"
  fi

  echo "[Phase 2] Rerun failed cases serially on Ascend device ${FALLBACK_ASCEND_DEVICE}"
  export ASCEND_RT_VISIBLE_DEVICES="${FALLBACK_ASCEND_DEVICE}"
  set +e
  pytest --lf --last-failed-no-failures=none -n 0 -v . --reruns 0
  serial_rc=$?
  set -e

  if [ "${serial_rc}" -ne 0 ]; then
    echo "[ERROR] Serial rerun still has failures on Ascend device ${FALLBACK_ASCEND_DEVICE}"
    return 1
  fi

  echo "[SUCCESS] Cases run passed on Ascend device ${FALLBACK_ASCEND_DEVICE}"
  return 0

}

pytestcase_dir=("passed_tests")
for test_dir in "${pytestcase_dir[@]}"; do
    echo "run pytestcase in ${test_dir}"
    run_pytestcases "${test_dir}"
done
