from typing import List
from triton.language.core import (
    block_type,
    tensor,
)
from triton._C.libtriton import ir

def insert_slice(ful: tensor, sub: tensor, offsets: List[tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = block_type(ful.type.scalar, ful.shape)
    out = builder.create_insert_slice(ful.handle, sub.handle, new_offsets, sizes, strides)
    return tensor(out, ret_type)

def extract_slice(ful: tensor, offsets: List[tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = block_type(ful.type.scalar, sizes)
    out = builder.create_extract_slice(ful.handle, new_offsets, sizes, strides)
    return tensor(out, ret_type)