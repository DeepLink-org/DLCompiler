#!/bin/bash

set -euo pipefail
script=$(readlink -f "$0")
script_dir=$(dirname "$script")

function run_pytestcases() {
  if [ -d ${HOME}/.triton/dump ]; then
    rm -rf ${HOME}/.triton/dump
  fi
  if [ -d ${HOME}/.triton/cache ]; then
    rm -rf ${HOME}/.triton/cache
  fi

  cd ${script_dir}
  TARGET_DIR="$1"
  cd ${TARGET_DIR}

  echo "[Phase 1] Run tests in parallel"
  set +e
  timeout --signal=TERM 40m  pytest . -n 8 --dist=loadscope --reruns 5 --reruns-delay 5
  parallel_rc=$?
  set -e

  if [ "${parallel_rc}" -eq 0 ]; then
    echo "[Phase 1] All tests passed"
    return 0
  fi

  echo "[Phase 2] Parallel run failed, rerun failed cases serially"
  set +e
  pytest --lf --last-failed-no-failures=none -n 0 -v .
  serial_rc=$?
  set -e

  if [ "${serial_rc}" -ne 0 ]; then
    echo "[Result] Serial rerun still has failures"
    return 1
  fi

  echo "[Result] Failed cases from parallel run passed in serial rerun"
  return 0

}

pytestcase_dir=("passed_tests")
for test_dir in "${pytestcase_dir[@]}"; do
    echo "run pytestcase in ${test_dir}"
    run_pytestcases "${test_dir}"
done
