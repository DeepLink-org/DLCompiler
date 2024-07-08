
import os
from typing import Any
from collections import namedtuple
import triton.backends.dicp_triton.libtriton.triton as _triton
from triton.backends.dicp_triton.libtriton.triton import ir
from triton._C.mlu_utils import get_device_properties
import torch_mlu
from pathlib import Path
import tempfile

def set_num_warps(mod: Any, num_warps: int):
    '''
    Set num warps on triton module.
    :param: mod: tt ir module.
    :num_warps: num warps set by user, it will attach attributes
                triton.xpe = num_warps > 1 ? 4 : 1 and
                triton.xtask = num_warps / 4.
    '''
    if num_warps not in [1, 4, 8, 16, 32]:
        raise ValueError("num_warps only support 1/4/8/16/32")

    builder = ir.builder(mod.context)
    if num_warps > 1:
        mod.set_attr("triton.xpe", builder.get_int32_attr(4))
        mod.set_attr("triton.xtask", builder.get_int32_attr(num_warps // 4))
    else:
        mod.set_attr("triton.xpe", builder.get_int32_attr(1))

def set_num_stages(mod: Any, num_stages: int):
    builder = ir.builder(mod.context)
    mod.set_attr("triton.num_stages", builder.get_int32_attr(num_stages))

def set_linear_attr(mod: Any, is_linear: bool):
    if is_linear is not None:
        builder = ir.builder(mod.context)
        mod.set_attr("triton.is_linear", builder.get_bool_attr(is_linear))

def set_debug_attr(mod: Any):
    debug = os.environ.get("TRITON_DEBUG", "0")
    if debug not in ["0", "1"]:
        raise ValueError("TRITON_DEBUG must be '0' or '1', but got {}", debug)
    if debug == "1":
        builder = ir.builder(mod.context)
        mod.set_attr("mlu.debug", builder.get_bool_attr(True))

def set_kernel_sym(mod: Any, kernel_name: str):
    if kernel_name is not None:
        builder = ir.builder(mod.context)
        mod.set_attr("triton.kernel_name", builder.get_str_attr(kernel_name))

def set_silence(mod: Any):
    builder = ir.builder(mod.context)
    mod.set_attr("mlu.silence", builder.get_bool_attr(True))

def ttir_get_kernel_name(ttir: str):
    '''
    Get kernel name from ttir.
    This Kernel name is required when launching the kernel.
    '''
    for line in ttir.split('\n'):
        line = line.strip()
        if line.startswith('tt.func'):
            return line.split('@')[1].split("(")[0]
    return None


def ttir_to_cnfatbin(mod: Any, arch_desc: dict, tuning_mode: bool, is_linear: bool):
    '''
    Compile triton module to cnfatbin.
    :param mod: tt ir module.
    :param arch_desc: contains num_warps and arch.
    :return: str
    '''
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "temp_ttir.mlir")
        ttir_code = str(mod)
        kernel_name = ttir_get_kernel_name(ttir_code)
        print(kernel_name)
        Path(src_path).write_text(ttir_code)
        context = ir.context()
        context.load_triton()
        mod = ir.parse_mlir_module(src_path, context)

        context2 = ir.context()
        context2.load_triton()
        builder = ir.builder(context2)
        builder.target = arch_desc
        mod2 = builder.create_module()
        if kernel_name is None:
            mod2.push_back(mod.get_single_function())
        else:
            mod2.push_back(mod.get_function(kernel_name))
        mod2.context = context2

        set_num_warps(mod2, arch_desc["num_warps"])
        set_num_stages(mod2, arch_desc["num_stages"])
        set_linear_attr(mod2, is_linear)
        set_debug_attr(mod2)
        set_kernel_sym(mod2, kernel_name)
        if tuning_mode:
            set_silence(mod2)
        return _triton.compile_ttir_to_cnfatbin(mod2, arch_desc["isa_version"])

def get_architecture_descriptor(device, **kwargs):
    # Currently only support Block task.
    num_warps = kwargs.get("num_warps", 1)
    num_stages = kwargs.get("num_stages", 1)
    isa_version = get_device_properties(device).get('isa_version')
    isa_version_passed = kwargs.get("isa_version", None)
    isa_version = isa_version if isa_version_passed is None else isa_version_passed
    capability = {
        "num_warps": num_warps,
        "isa_version": isa_version,
        "num_stages" : num_stages,
    }
    return capability