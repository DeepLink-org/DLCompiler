from pathlib import Path
import tempfile
import os
import re
import subprocess
import sysconfig
import functools
import hashlib
from triton.runtime.cache import get_cache_manager, get_dump_manager
from triton.backends.compiler import GPUTarget

from .utils import (
    TRITON_PROFILER_REGISTERED,
    _check_cxx11_abi,
    _get_ascend_path,
    _get_bisheng_path,
    _build_npu_ext,
    _precompile_npu_hash,
    _precompile_npu_ext,
    _precompile_npu_ext_with_lock,
    _is_auto_map_parallel_blocks_enabled,
    convert_sigtype_to_int,
    get_ascend_arch_from_env,
    is_ffts_supported,
    force_disable_ffts,
    get_backend_func,
)


# ---------------------------------------------------------------------------
# NPUUtils — singleton that loads npu_utils.cpp
# ---------------------------------------------------------------------------


class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "npu_utils.cpp")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "npu_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "npu_utils.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_npu_ext("npu_utils", None, src_path)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util

        spec = importlib.util.spec_from_file_location("npu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod

    def load_binary(self, name, kernel, shared, device, mix_mode=None):
        if mix_mode is None:
            if " " in name:
                fnname, mix_mode = name.split()
            else:
                fnname = name
                mix_mode = "aiv"
        else:
            fnname = name
        return self.npu_utils_mod.load_kernel_binary(
            fnname, kernel, shared, device, mix_mode
        )

    @functools.lru_cache()
    def get_device_properties(self, device):
        num_aic = self.get_aicore_num()
        num_aiv = num_aic * 2
        return {"max_shared_mem": 1, "num_aicore": num_aic, "num_vectorcore": num_aiv}

    @functools.lru_cache()
    def get_arch(self):
        return self.npu_utils_mod.get_arch()

    @functools.lru_cache()
    def get_aicore_num(self):
        return self.npu_utils_mod.get_aicore_num()

    @functools.lru_cache()
    def get_aivector_core_num(self):
        return self.get_device_properties("npu")["num_vectorcore"]


# ---------------------------------------------------------------------------
# Header source generation (precompiled separately from wrapper)
# ---------------------------------------------------------------------------


def generate_npu_header_src():
    enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", "true").lower() in (
        "true",
        "1",
    )
    return f"""
#ifndef TRITON_NPU_HEADERS
#define TRITON_NPU_HEADERS

#include <assert.h>
#include <stdbool.h>
#include <string>
#include <sys/syscall.h>
#include <vector>
#include <Python.h>
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
{get_backend_func("header_file", enable_taskqueue)}

#endif
"""


# ---------------------------------------------------------------------------
# Device-print code extraction from CANN ccelib
# ---------------------------------------------------------------------------


def extract_device_print_code_from_cann():
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(_get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(
        ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib"
    )

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), "r") as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = " ".join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = "\n".join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer(r"\[aicore\]", code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos - 1, -1, -1):
                if src[i] == "}":
                    left = i + 1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == "{":
                    brace_nest += 1
                elif src[j] == "}":
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ""
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += code[segment_start:]

        new_code = new_code.replace("__gm__", " ")
        new_code = new_code.replace("__CCELIB_RT_ERROR_NONE", "RT_ERROR_NONE")
        new_code = new_code.replace("__CCELIB_RT_MEMORY_HBM", "RT_MEMORY_HBM")
        new_code = new_code.replace(
            "__CCELIB_RT_MEMCPY_HOST_TO_DEVICE", "RT_MEMCPY_HOST_TO_DEVICE"
        )
        new_code = new_code.replace(
            "__CCELIB_RT_MEMCPY_DEVICE_TO_HOST", "RT_MEMCPY_DEVICE_TO_HOST"
        )
        return new_code

    headers_combined = "\n".join(
        [
            read_header("common/common_impl.h"),
            read_header("internal/debug_tunnel/payload.h"),
            read_header("internal/debug_tunnel/payload_impl.h"),
            read_header("internal/debug_tunnel/tunnel.h"),
            read_header("internal/debug_tunnel/tunnel_impl.h"),
        ]
    )
    return "#include <iostream>\n" + headers_combined


# ---------------------------------------------------------------------------
# Type helpers (merged _ty_to_cpp + _extracted_ty)
# ---------------------------------------------------------------------------


def ty_to_cpp(ty):
    if ty[0] == "*":
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def extracted_ty(ty):
    if ty[0] == "*":
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return ty_to_cpp(ty)


def format_of(ty):
    if ty[0] == "*":
        return "O"
    if ty == "constexpr":
        return "O"
    return {
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "L",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty_to_cpp(ty)]


def _format_of_msprof_task_type_ratio(bs_task_type, mix_mode):
    default_task_type = (
        "MSPROF_GE_TASK_TYPE_AIV"
        if mix_mode == "aiv"
        else "MSPROF_GE_TASK_TYPE_AI_CORE"
    )
    if not bs_task_type:
        return default_task_type, 0
    task_type_num, mix_block_dim_ratio = divmod(int(bs_task_type), 10)
    task_type_map = {
        1: "MSPROF_GE_TASK_TYPE_AIV",
        2: "MSPROF_GE_TASK_TYPE_AI_CORE",
        3: "MSPROF_GE_TASK_TYPE_MIX_AIC",
        4: "MSPROF_GE_TASK_TYPE_MIX_AIV",
    }
    return task_type_map.get(task_type_num, default_task_type), mix_block_dim_ratio


# ---------------------------------------------------------------------------
# Wrapper source generation
# ---------------------------------------------------------------------------


def generate_npu_wrapper_src(
    constants,
    signature,
    workspace_size,
    mix_mode,
    lock_num,
    lock_init_value,
    *,
    bs_task_type=0,
    compile_on_910_95=False,
    force_simt_only=False,
    shared_mem_dynamic_size=0,
):
    import os

    # TODO(zmz): temporary workaround — signature values with *u1 should become *i1
    signature = {k: v.replace("*u1", "*i1") for k, v in signature.items()}

    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ",".join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ",".join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == "*":
            return "PyObject*"
        if ty == "constexpr":
            return "PyObject*"
        return ty_to_cpp(ty)

    def _format_of(ty):
        if isinstance(ty, tuple):
            val = "".join(map(_format_of, ty))
            return f"({val})"
        if ty[0] == "*":
            return "O"
        if ty == "constexpr":
            return "O"
        if ty == "void*":
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    # Compute args_format BEFORE re-indexing (needs original signature types)
    args_format = "".join([_format_of(ty) for ty in signature.values()])
    fmt = "iiiKKOOOO" + args_format

    # Serialize and re-index signature: flatten tuples, re-number from 0
    signature = ",".join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(",")))
    signature = {i: s for i, s in enumerate(signature)}

    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )

    arg_decls = ", ".join(
        f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr"
    )
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    grid_info = {"X": "i32", "Y": "i32", "Z": "i32"}

    newline = "\n  "
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]

    enable_device_print = os.getenv("TRITON_DEVICE_PRINT", "false").lower() in (
        "true",
        "1",
    )
    enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", "true").lower() in (
        "true",
        "1",
    )
    enable_grid_warn_print = os.getenv("TRITON_GRID_WARN_PRINT", "false").lower() in (
        "true",
        "1",
    )
    enable_auto_map_parallel_blocks = _is_auto_map_parallel_blocks_enabled()
    npu_utils = NPUUtils()
    num_physical_blocks = (
        npu_utils.get_aivector_core_num()
        if mix_mode == "aiv"
        else npu_utils.get_aicore_num()
    )

    task_type, mix_block_dim_ratio = _format_of_msprof_task_type_ratio(
        bs_task_type, mix_mode
    )
    is_mix_task_type = "true" if ("MIX" in task_type) else "false"
    LINE_CHANGE_CHAR = chr(10)
    alloc_success_code = "return 1;"
    sync_lock_fail_code = (
        'fprintf(stderr, "Error: syncBlockLock allocation failed\\n"); return;'
    )
    workspace_fail_code = (
        'fprintf(stderr, "Error: workspace allocation failed\\n"); return;'
    )

    enable_simt = ("simt" in mix_mode) or force_simt_only

    arch = get_ascend_arch_from_env()
    target_support_ffts = is_ffts_supported(arch) and (not force_disable_ffts())

    cpp_device_pointer = """
typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    aclrtPtrAttributes attributes;
    aclError status = aclrtPointerGetAttributes(ptr_info.dev_ptr, &attributes);
    if (status == ACL_SUCCESS) {
      if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE && attributes.location.type != 4) {
        Py_DECREF(ret);
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
        return ptr_info;
      }
    } else {
      Py_DECREF(ret);
      PyErr_Format(PyExc_RuntimeError,
                   "Failed to query pointer attributes at argument %d. "
                   "Error code: %d. This may indicate invalid memory address "
                   "or NPU device error.",
                   idx, status);
      ptr_info.valid = false;
      return ptr_info;
    }
    Py_DECREF(ret);
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}
"""

    cpp_msprof_extern = """
extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)
        return 1;
      if (handle->type == 1) {
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}
"""

    cpp_msprof_callback = """
  MsprofRegisterCallback(8, ProfCtrlHandle);
"""

    cpp_msprof_call_before_launch = """
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }
"""

    cpp_msprof_call_after_launch = f"""
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {{
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }}
    if (__MsprofFlagL1)
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = {task_type};
      nodeBasicInfo.data.nodeBasicInfo.blockDim = nodeBasicBlockDim;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // 'mix' kernel need to report the ctxID
      if ({is_mix_task_type} > 0) {{
        MsprofAdditionalInfo info;
        info.level = MSPROF_REPORT_NODE_LEVEL;
        info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
        info.threadId = threadId;
        info.timeStamp = endTime;
        MsprofContextIdInfo ctxId;
        ctxId.opName = opNameHashID;
        ctxId.ctxIdNum = 1;
        for (uint32_t i = 0; i < ctxId.ctxIdNum; i++) {{
          ctxId.ctxIds[i] = i;
        }}
        size_t copyLen = sizeof(MsprofContextIdInfo);
        if (copyLen > MSPROF_ADDTIONAL_INFO_DATA_LENGTH) {{
          copyLen = MSPROF_ADDTIONAL_INFO_DATA_LENGTH;
        }}
        memcpy(info.data, &ctxId, copyLen);
        MsprofReportAdditionalInfo(false, static_cast<void *>(&info), sizeof(MsprofAdditionalInfo));
      }}

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {{
        {newline.join(
          f'dataTypes[{i}] = {convert_sigtype_to_int(ty[1:])};'
          for i, ty in signature.items()
          if ty[0] == "*" and i < 5
        )}
      }}
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {{
        auto fillTensorData = [&](int index, int tensorType) {{
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2;
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {{
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }}
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {{
            profTensorData->tensorData[index].shape[j] = 0;
          }}
        }};
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }}
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }}
      }}
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }}
"""

    # Kernel launch: SIMT path for 910_95
    cpp_kernel_launch = f"""
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);
"""
    if compile_on_910_95 and enable_simt:
        cpp_kernel_launch = f"""
    rtArgsEx_t argsInfo = {{}};
    argsInfo.args = static_cast<void*>(&args);
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo = {{}};
    cfgInfo.localMemorySize = {shared_mem_dynamic_size};
    ret = rtKernelLaunchWithFlagV2(func, blockNum, &argsInfo, NULL, stream, 0, &cfgInfo);
"""

    precompile_headers = """
#include "precompiled.h"
"""

    return f"""
{precompile_headers}
{'#define __CCE_ENABLE_PRINT__' if enable_device_print else ''}
{extract_device_print_code_from_cann() if enable_device_print else ''}
#define PY_SSIZE_T_CLEAN
{'#define ENABLE_GRID_WARN_PRINT' if enable_grid_warn_print else ''}
#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

{cpp_msprof_extern}

{cpp_device_pointer}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds{', ' + arg_decls if len(signature) > 0 else ''}) {{
  std::string name = "";
  name.append(kernelName);
  void *workspace_addr_ptr = NULL;
  uint32_t blockNum4Workspace = gridX * gridY * gridZ;
  {get_backend_func("pre_launch", True)}
  {f'''
  uint64_t totalWorkSpaceSize = {workspace_size} * blockNum4Workspace;
  {get_backend_func("allocate_memory", "totalWorkSpaceSize", "stream")}
  ''' if workspace_size > 0 else ''}
  {'auto launch_call = [=]() -> rtError_t' if enable_taskqueue else ''} {{
    {get_backend_func("pre_launch", False)}
    uint32_t blockNum = gridX * gridY * gridZ;
    #ifdef ENABLE_GRID_WARN_PRINT
      static bool warned = false;
      if (!warned && blockNum > (uint32_t){num_physical_blocks}) {{
        printf("WARNING: Grid %u > physical limit {num_physical_blocks}, performance maybe reduced.\\n",blockNum);
        warned = true;
    }}
    #endif

    {'blockNum = std::min(blockNum, (uint32_t)' + str(num_physical_blocks) + ');' if enable_auto_map_parallel_blocks else ''}
    // set mixBlockDimRatio for nodeBasicBlockDim for msprof report
    uint32_t mixBlockNumRation = {mix_block_dim_ratio};
    uint32_t nodeBasicBlockDim = (mixBlockNumRation << 16) + blockNum;

    {'cce::internal::DebugTunnelData *DTData = cce::internal::DebugTunnel::Open(blockNum);' if enable_device_print else ''}
    rtError_t ret = RT_ERROR_NONE;
    {'void *ffts_addr = NULL; uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);' if target_support_ffts else ''}
    {'if (ret != RT_ERROR_NONE) return ret;' if (target_support_ffts and enable_taskqueue) else 'if (ret != RT_ERROR_NONE) return;' if (target_support_ffts and (not enable_taskqueue)) else ''}
    // stub argument for syncBlockLock
    void *syncBlockLock_ptr = NULL;
    uint16_t ModuleId = 0;
    {f'''
    uint64_t syncBlockLockSize = {lock_num} * sizeof(int64_t);
    {get_backend_func("allocate_sync_block_lock", "syncBlockLockSize", "stream")}
    if (!syncBlockLock_ptr) {{
      {alloc_success_code if enable_taskqueue else sync_lock_fail_code}
    }}
    std::vector<int64_t> lockInitData({lock_num}, {lock_init_value});
    ret = rtMemcpy(syncBlockLock_ptr, syncBlockLockSize, reinterpret_cast<void *>(lockInitData.data()),
                   syncBlockLockSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    ''' if lock_num > 0 else ''}
    struct __attribute__((packed)) {{
      {'void* ffts_addr __attribute__((aligned(8)));' if target_support_ffts else ''}
      {'void* syncBlockLock __attribute__((aligned(8)));' if not force_simt_only else ''}
      {'void* workspace_addr __attribute__((aligned(8)));' if not force_simt_only else ''}
      {' '.join(f'{ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if i not in constants and ty != "constexpr")}
      {' '.join(f'{ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items() if ty != "constexpr")}
      {'void* DTData __attribute__((aligned(8)));' if enable_device_print else ''}
    }} args = {{
      {'static_cast<void*>(ffts_addr),' if target_support_ffts else ''}
      {('static_cast<void*>(syncBlockLock_ptr),' if lock_num > 0 else 'nullptr,') if not force_simt_only else ''}
      {('static_cast<void*>(workspace_addr_ptr),' if workspace_size > 0 else 'nullptr,') if not force_simt_only else ''}
      {(lambda _rt: (', '.join(_rt) + ',') if _rt else '')(
        [f'static_cast<{ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if i not in constants and ty != "constexpr"]
      )}
      {', '.join(f'static_cast<{ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items() if ty != "constexpr")}
      {', static_cast<void*>(DTData)' if enable_device_print else ''}
    }};
    {cpp_msprof_call_before_launch}
    {cpp_kernel_launch}
    {'void *&stream_ref = const_cast<void*&>(stream);' if enable_device_print else ''}
    {'cce::internal::DebugTunnel::Close(DTData, stream_ref);' if enable_device_print else ''}
    {cpp_msprof_call_after_launch}
    {'return ret;' if enable_taskqueue else 'ret = rtStreamSynchronize(stream);'}
   }};
   {f'''{get_backend_func("async_launch", "launch_call") if enable_taskqueue else ''}'''}
  return;
}}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_shape(PyObject *tensor) {{
  std::vector<int64_t> shape;

  if (!tensor || tensor == Py_None) {{
    return shape;
  }}

  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {{
    return shape;
  }}
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {{
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {{
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {{
        shape.push_back(PyLong_AsLong(dim));
      }}
    }}
  }}
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;
  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{fmt}\",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook{args_list})) {{
    return NULL;
  }}
  if (__MsprofFlagL1)
  {{
    {
      newline.join(
        f"{{ auto tmp = _get_tensor_shape(_arg{i}); if (!tmp.empty()) tensorShapes.push_back(tmp); }}"
        for i, ty in signature.items() if ty[0] == "*"
      )
    }
  }}

  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {{
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {{
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }}
  }}

  // raise exception asap
  {newline.join(ptr_decls)}
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}
  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL,
  -1,
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  {cpp_msprof_callback}
  return m;
}}
"""


# ---------------------------------------------------------------------------
# Launcher stub builder
# ---------------------------------------------------------------------------


def make_npu_launcher_stub(header_src, wrapper_src, debug=False):
    enable_precompile = not os.getenv("TRITON_DISABLE_PRECOMPILE", "false").lower() in (
        "true",
        "1",
    )
    header_path = _precompile_npu_ext_with_lock(header_src, enable_precompile)
    assert header_path is not None, "the precompiled.h path is empty."

    so_cache_key = hashlib.sha256(wrapper_src.encode("utf-8")).hexdigest()
    so_cache_manager = get_cache_manager(so_cache_key)
    use_cxx11_abi = _check_cxx11_abi()
    name = f"launcher_cxx11abi{use_cxx11_abi}"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_name = f"{name}{suffix}"

    if debug:
        dump_manager = get_dump_manager(so_cache_key)
        if header_path is not None:
            print(f"Dumping precompiled.h to {dump_manager.cache_dir}")
            dump_manager.put(header_src, "precompiled.h", binary=False)
        print(f"Dumping {name}.cxx to {dump_manager.cache_dir}")
        dump_manager.put(wrapper_src, f"{name}.cxx", binary=False)

    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is not None:
        return cache_path

    kernel_launcher_type = "torch"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"{name}.cxx")
        with open(src_path, "w") as f:
            f.write(wrapper_src)
        so_path = _build_npu_ext(
            name,
            header_path,
            src_path,
            kernel_launcher=kernel_launcher_type,
            precompile=enable_precompile,
        )
        if debug:
            with open(so_path, "rb") as f:
                return dump_manager.put(f.read(), so_name, binary=True)
        with open(so_path, "rb") as f:
            return so_cache_manager.put(f.read(), so_name, binary=True)


# ---------------------------------------------------------------------------
# NPULauncher
# ---------------------------------------------------------------------------


class NPULauncher(object):
    def __init__(self, src, metadata):
        self.compile_only = os.getenv("TRITON_COMPILE_ONLY", "false").lower() in (
            "true",
            "1",
        )
        self.enable_msprof_register_tensor = os.getenv(
            "TRITON_REGISTER_TENSOR_MSPROF", "false"
        ).lower() in ("true", "1")
        debug_mode = metadata.debug
        workspace_size = (
            int(metadata.workspace_size) if hasattr(metadata, "workspace_size") else -1
        )
        lock_init_value = (
            int(metadata.lock_init_value) if hasattr(metadata, "lock_init_value") else 0
        )
        lock_num = int(metadata.lock_num) if hasattr(metadata, "lock_num") else -1
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        mix_mode = metadata.mix_mode

        bs_task_type = getattr(metadata, "bs_task_type", 0)
        compile_on_910_95 = getattr(metadata, "compile_on_910_95", False)
        force_simt_only = getattr(metadata, "force_simt_only", False)
        shared_mem_dynamic_size = getattr(metadata, "shared_mem_dynamic_size", 0)

        header_src = generate_npu_header_src()
        wrapper_src = generate_npu_wrapper_src(
            constants,
            signature,
            workspace_size,
            mix_mode,
            lock_num,
            lock_init_value,
            bs_task_type=bs_task_type,
            compile_on_910_95=compile_on_910_95,
            force_simt_only=force_simt_only,
            shared_mem_dynamic_size=shared_mem_dynamic_size,
        )
        self.so_launcher_path = make_npu_launcher_stub(
            header_src, wrapper_src, debug_mode
        )
        self.mix_mode = mix_mode
        self.shared = metadata.shared if hasattr(metadata, "shared") else 1

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "__triton_launcher", self.so_launcher_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")

    def __call__(self, *args, **kwargs):
        if self.compile_only:
            cache_manager = get_cache_manager(args[5]["hash"])
            print("[INFO]: skip running kernel")
            print(f"[INFO]: The compiled kernel cache is in {cache_manager.cache_dir}")
        if self.enable_msprof_register_tensor:
            # args[5] must be the packed metadata
            args = list(args)
            args[5]["tensor_params_shape"] = get_backend_func(
                "get_tensor_params_shape", *args
            )
        else:
            if self.compile_only:
                return
            profiler_registered = self.launch(*args, **kwargs)
            from . import utils as _backend_utils

            _backend_utils.TRITON_PROFILER_REGISTERED = (
                True if profiler_registered == 1 else False
            )
