#!/bin/bash
function usage() {
    echo
    echo "Usage: $(basename $0) [options ...]"
    echo
    echo "Options:"
    echo " -k|--keep_cache"
    echo " -t|--test"
    echo " -w|--wheel"
    echo " -b|--build_type <Release|Debug>"
    echo " -l|--llvm <llvm_path>"
    echo " -m|--maca <maca_path>"
    echo " -e|--editable"
    echo " -d|--disable-dist"
    echo " -h|--help"
    return 0
}

function err_msg() {
    echo "Error when parsing args. Abort"
}

function main() {
    local cur_dir="$(cd "$(dirname "$0")" ; pwd -P)"
    local triton_dir=${cur_dir}/../../
    local python_triton_dir=${triton_dir}/

    local pybind11_release_dir=${triton_dir}/third_party/pybind11/pybind11-2.11.1/
    local json_release_dir=${triton_dir}/third_party/json/

    local extmathlib_dir=${triton_dir}/third_party/mcExtMathLib/
    local extmathlib_install_dir=${triton_dir}/third_party/metax/backend/
    local extmathlib_bc=${triton_dir}/third_party/metax/backend/lib/ext_maca_mathlib.bc

    # config
    local CLEAN_CACHE=1
    local RUN_TEST=0
    local BUILD_WHEEL=0
    local BUILD_TYPE=Release
    local BUILD_DIST=1
    local EDITABLE=0
    local LLVM_RELEASE_DIR=${triton_dir}/third_party/llvm_release/
    local MACA_PATH="/opt/maca/"

    while [ -n "$1" ]; do
        case "$1" in
            -k|--keep_cache)
                CLEAN_CACHE=0
                shift 1
                ;;
            -t|--test)
                RUN_TEST=1
                shift 1
                ;;
            -w|--wheel)
                BUILD_WHEEL=1
                shift 1
                ;;
            -b|--build_type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -l|--llvm)
                LLVM_RELEASE_DIR="$2"
                shift 2
                ;;
            -m|--maca)
                MACA_PATH="$2"
                shift 2
                ;;
            -e|--editable)
                EDITABLE=1
                shift 1
                ;;
            -d|--disable-dist)
                BUILD_DIST=0
                shift 1
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                err_msg
                usage
                return 1
                ;;
       esac
    done

    export MACA_PATH=${MACA_PATH}
    export LD_LIBRARY_PATH=${MACA_PATH}/lib/:${MACA_PATH}/mxgpu_llvm/lib/:${LD_LIBRARY_PATH}

    export TRITON_OFFLINE_BUILD=1
    export LLVM_SYSPATH=${LLVM_RELEASE_DIR}
    export LLVM_LIBRARY_DIR=${LLVM_RELEASE_DIR}/lib
    export LLVM_INCLUDE_DIRS=${LLVM_RELEASE_DIR}/include
    export PYBIND11_SYSPATH=${pybind11_release_dir}
    export JSON_SYSPATH=${json_release_dir}
    export THIRDPARTY_MANUAL=1
    export TRITON_BUILD_PROTON=OFF
    # for triton distributed build
    if [[ ${BUILD_DIST} == 1 ]]; then
        export TRITON_BUILD_DISTRIBUTED="ON"
        export CUDA_PATH=${MACA_PATH}/tools/cu-bridge
        export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
        export CUCC_CMAKE_ENTRY=2
        export PATH=${CUDA_PATH}/bin:${CUCC_PATH}/tools:${PATH}
    else
        echo "Turn off TRITON_BUILD_DISTRIBUTED"
        export TRITON_BUILD_DISTRIBUTED="OFF"
    fi
    if [[ ${BUILD_TYPE} =~ "debug" ]]; then
        export DEBUG=1
    fi

    if [[ ${CLEAN_CACHE} != 0 ]]; then
        rm -rf ${triton_dir}/triton.egg-info
        rm -rf ${triton_dir}/.pytest_cache
        rm -rf ${triton_dir}/tests/__pycache__
        rm -rf ${triton_dir}/tests/__pycache__
        rm -rf ${triton_dir}/dist/
        rm -rf ${triton_dir}/build
        rm -rf ${triton_dir}/maca/maca_tests/python_test/__pycache__
        rm -rf ${triton_dir}/maca/maca_tests/pytest_test/__pycache__
        rm -rf ${triton_dir}/maca/maca_tests/pytest_test/.pytest_cache
        rm -rf ${triton_dir}/maca/maca_tests/pytest_test/.tmp
    fi

    # build extmathlib
    if [ ! -f ${extmathlib_bc} ]; then
        cd $extmathlib_dir
        rm -rf build && mkdir build && cd build
        cmake -DCMAKE_INSTALL_PREFIX=${extmathlib_install_dir} ..
        make install
        if [ $? -ne 0 ]; then
            echo "mathlib build failed"
            exit 1
        fi
    fi

    # copy mlir-opt
    mlir_opt_path=$LLVM_SYSPATH/bin/mlir-opt
    backend_path=$triton_dir/third_party/metax/backend/bin/
    if [ ! -d $backend_path ]; then
        mkdir -p $backend_path
        if [ ! -e $mlir_opt_path ]; then
            echo "mlir-opt does not exist, please try to run build_llvm.sh firstly."
            exit 1
        fi
        cp $mlir_opt_path $backend_path
    fi

    # build and install
    cd ${python_triton_dir}
    pip install -r maca/maca_tools/requirements.txt
    if [[ ${BUILD_WHEEL} == 1 ]]; then
        python setup.py bdist_wheel
    else
        if [[ ${EDITABLE} == 0 ]]; then
            python -m pip install .
        else
            python -m pip install -e .
        fi
    fi

    # test
    ret1=0
    if [ ! -f "./test/lit.site.cfg.py" ]; then
        find ./ -name "lit.site.cfg.py" | xargs -i cp {} ./test/
        ret1=$?
    fi
    ret2=0
    if [[ ${RUN_TEST} == 1 ]]; then
        pip install -e '.[tests]'
        pytest -vs test/unit/
        ret2=$?
    fi
    if [[ $ret1 == 0 && $ret2 == 0 ]]; then
        ret=0
    else
        ret=1
    fi
    return ${ret}
}

main "$@"
exit $?
