#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR})

pushd ${PROJECT_ROOT}/../3rdparty/triton
MAX_JOBS=126 pip3 install -e python --verbose --no-build-isolation
popd
