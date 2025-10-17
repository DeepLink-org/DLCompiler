#!/bin/bash

set -ex
script=$(readlink -f "$0")
script_dir=$(dirname "$script")

function run_pytestcases() {
#   if [ -d ${HOME}/.triton/dump ]; then
#     rm -rf ${HOME}/.triton/dump
#   fi
#   if [ -d ${HOME}/.triton/cache ]; then
#     rm -rf ${HOME}/.triton/cache
#   fi

  cd ${script_dir}
  TARGET_DIR="$1"
  cd ${TARGET_DIR}
  pytest -n 16 --dist=load . || { exit 1 ; }

}

pytestcase_dir=("pytest_ut")
for test_dir in "${pytestcase_dir[@]}"; do
    echo "run pytestcase in ${test_dir}"
    run_pytestcases ${test_dir}
done



