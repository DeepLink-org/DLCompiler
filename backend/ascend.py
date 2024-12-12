import hashlib
import os
import tempfile
import shutil
import subprocess
import sysconfig
import contextlib
import sys
import io
import functools
import importlib
import setuptools
from pathlib import Path
from triton.runtime.cache import get_cache_manager
from triton.runtime import JITFunction
from .utils import quiet
import torch
from torch_npu.contrib import transfer_to_npu

def llir_to_ascendc(mod, metadata):
    src = f"""
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelCustom {{
public:
    __aicore__ inline KernelCustom() {{}}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {{
        this->blockLength = totalLength / GetBlockNum(); 
        this->tileNum = 8;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + this->blockLength * GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }}
    __aicore__ inline void Process()
    {{
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {{
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {{
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }}
    __aicore__ inline void Compute(int32_t progress)
    {{
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }}
    __aicore__ inline void CopyOut(int32_t progress)
    {{
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
}};

extern "C" __global__ __aicore__ void {metadata['name']}(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
{{
    KernelCustom op;
    op.Init(x, y, z, totalLength);
    op.Process();
}}
    """
    return src


def generate_pybind_cpp(name):
    src = f"""
#include <pybind11/pybind11.h>
#include "aclrtlaunch_{name}.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

PYBIND11_MODULE(__triton_launcher, m) {{
    m.def("launch", &aclrtlaunch_{name}, "");
}}
    """
    return src


def generate_cmakelist(soc_version, cann_path, name, mode='npu'):
    src = f"""
cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# user-defined configuration
set(SOC_VERSION "{soc_version}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "{cann_path}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "{mode}" CACHE STRING "run mode: npu/sim/cpu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${{CMAKE_CURRENT_LIST_DIR}}/out" CACHE STRING "path for install()" FORCE)

if(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${{ASCENDC_CMAKE_DIR}}/ascendc.cmake)

ascendc_library(kernels STATIC
    {name}.cpp
)

add_library(pybind11_lib SHARED pybind11.cpp)
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch_npu
)
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${{TORCH_PATH}}")
set(ENV{{ASCEND_HOME_PATH}} ${{ASCEND_CANN_PACKAGE_PATH}})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${{TORCH_NPU_PATH}}")
target_link_directories(pybind11_lib PRIVATE
  ${{TORCH_PATH}}/lib
  ${{TORCH_NPU_PATH}}/lib
)
target_include_directories(pybind11_lib PRIVATE
  ${{TORCH_NPU_PATH}}/include
  ${{TORCH_PATH}}/include
  ${{TORCH_PATH}}/include/torch/csrc/api/include
)
execute_process(COMMAND python3 -m pybind11 --includes 
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${{PYBIND11_INC}})
target_compile_options(pybind11_lib PRIVATE
  ${{PYBIND11_INC}}
  -D_GLIBCXX_USE_CXX11_ABI=0
)

execute_process(COMMAND python3-config --extension-suffix
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME {name}${{PYBIND11_SUFFIX}}
  PREFIX "" SUFFIX ""
)
    """
    return src


def generate_launcher_so(src, metadata):
    current_dir = os.getcwd()
    name = metadata['name']
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"{name}.cpp")
        with open(src_path, "w") as f:
            f.write(src)
        pybind_src = generate_pybind_cpp(name)
        pybind_src_path = os.path.join(tmpdir, "pybind11.cpp")
        with open(pybind_src_path, "w") as f:
            f.write(pybind_src)
        #soc_version = torch.cuda.get_device_properties(0)['name']
        soc_version = "Ascend910B2"
        cann_path = os.getenv('ASCEND_TOOLKIT_HOME')
        cmake_src = generate_cmakelist(soc_version, cann_path, name)
        cmake_src_path = os.path.join(tmpdir, "CMakeLists.txt")
        with open(cmake_src_path, "w") as f:
            f.write(cmake_src)
        build_dir = os.path.join(tmpdir, "build")
        if os.path.exists(build_dir):
            os.removedirs(build_dir)
        os.makedirs(build_dir)
        os.chdir(build_dir)
        subprocess.check_call(["cmake", ".."])
        subprocess.check_call(["make"])

        so_cache_manager = get_cache_manager(metadata['hash'])
        files = os.listdir(build_dir)
        for file in files:
            if file.endswith(".so"):
                so_file = os.path.join(build_dir, file)
                break
        with open(so_file, "rb") as f:
            cache_path = so_cache_manager.put(f.read(), f"{name}.so", binary=True)
    os.chdir(current_dir)
    return cache_path


def load_binary(name, kernel, shared, device):
    return None, None, 0, 0


class AscendUtils(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AscendUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.load_binary = load_binary
        self.get_device_properties = torch.cuda.get_device_properties


class AscendLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        self.signature = signature
        self.constants = constants
        self.hash_key = metadata['hash']
        self.name = metadata['name']

    def __call__(self, grid_0, grid_1, grid_2, stream, kernel_function, kernel_packed_metadata, launch_metadata, launch_enter_hook,launch_exit_hook, *args, **kwargs):
        so_cache_manager = get_cache_manager(self.hash_key)
        cache_path = so_cache_manager.get_file(f"{self.name}.so")
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        spec = importlib.util.spec_from_file_location("__triton_launcher", cache_path)
        mod = importlib.util.module_from_spec(spec)
        params = tuple([args[i] for i in self.signature.keys() if i not in self.constants])
        mod.launch(grid_0, stream, *params)
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)