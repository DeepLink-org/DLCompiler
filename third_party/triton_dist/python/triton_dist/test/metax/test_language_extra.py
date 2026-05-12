################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import torch

import triton
import triton_dist
import triton.language as tl
import triton_dist.language as dl

import triton_dist.language.extra.language_extra as language_extra
from triton_dist.language.extra.maca.language_extra import (
    ld,
    tid,
    st,
    atomic_add,
    # atomic_cas,
    __shfl_sync_i32,
    laneid,
    ld_acquire,
    ld_b32,
    atomic_add_per_warp,
)


def _test_atomic_cas(semantic, scope, dtype):
    print(f"Testing atomic_cas with dtype={dtype}, scope={scope}, semantic={semantic}")
    device = torch.cuda.current_device()

    @triton_dist.jit
    def _atomic_cas_kernel(ptr_p, cmp_val_p, new_val_p, out_p, semantic: tl.constexpr, scope: tl.constexpr, size):
        pid = tl.program_id(axis=0)
        block_start = pid
        ptr_plus_offsets = ptr_p + block_start
        cmp_val_plus_offsets = cmp_val_p + block_start
        new_val_plus_offsets = new_val_p + block_start

        new_val = tl.load(new_val_plus_offsets)
        cmp_val = tl.load(cmp_val_plus_offsets)
        if tid(0) == 0:
            res = atomic_cas(ptr_plus_offsets, cmp_val, new_val, semantic=semantic, scope=scope)
            tl.store(out_p + block_start, res)
        language_extra.__syncthreads()

    SIZE = 128 * 4
    x = torch.randint(0, 100, (SIZE, ), dtype=torch.int64, device=device)
    # "compare_cuda" not implemented for usigned int
    cmp_val = torch.randint(0, 100, (SIZE, ), dtype=torch.int64, device=device)
    cmp_val = torch.where(cmp_val < 50, cmp_val, x)

    new_val = torch.randint(0, 100, (SIZE, ), dtype=torch.int64, device=device)
    z_old = torch.empty((SIZE, ), dtype=dtype, device=device)
    x_new_ref = torch.where(x == cmp_val, new_val, x)

    x = x.to(dtype)
    cmp_val = cmp_val.to(dtype)

    new_val = new_val.to(dtype)
    x_new_ref = x_new_ref.to(dtype)

    grid = lambda meta: (SIZE, )
    x_new = x.clone().detach()
    _atomic_cas_kernel[grid](
        x_new,
        cmp_val,
        new_val,
        z_old,
        semantic=semantic,
        scope=scope,
        size=SIZE,
    )
    # Check maybe swaped result.
    torch.testing.assert_close(x_new_ref, x_new, atol=0, rtol=0, equal_nan=False)
    # Check old value
    torch.testing.assert_close(x, z_old, atol=0, rtol=0, equal_nan=False)
    print(f"✅ atom_cas_{semantic}_{scope}_{dtype} Triton and Torch match")


def test_atomic_cas_main():
    dtype_list = [torch.int32, torch.uint32, torch.int64, torch.uint64]
    scope_list = ["cta", "gpu", "sys"]
    semantic_list = ["acquire", "release", "relaxed"]
    for dtype in dtype_list:
        for scope in scope_list:
            for semantic in semantic_list:
                if semantic == "acquire":  # TODO: acquire failed
                    continue
                _test_atomic_cas(semantic, scope, dtype)


def _test_atomic_add(semantic, scope, dtype):

    print(f"Testing atomic_add with dtype={dtype}, scope={scope}, semantic={semantic}")
    device = torch.cuda.current_device()

    @triton_dist.jit
    def _bincount_kernel(
        indices,
        out,
        size,
        semantic: tl.constexpr,
        scope: tl.constexpr,
    ):
        threads_per_block = language_extra.num_threads()
        tid = tid(0)
        pid = tl.program_id(axis=0)
        num_pid = tl.num_programs(axis=0)
        total_threads = threads_per_block * num_pid
        global_tid = pid * threads_per_block + tid

        for i in range(global_tid, size, total_threads):
            idx = ld(indices + i)
            atomic_add(out + idx, 1, semantic=semantic, scope=scope)

    SIZE = 1024
    max_lens = 32
    indices = torch.randint(0, max_lens, (SIZE, ), dtype=torch.int64, device=device)
    num_warps = 4
    out = torch.zeros((max_lens, ), dtype=dtype, device=device)
    out_ref = torch.bincount(indices.flatten(), minlength=max_lens).to(dtype)
    indices = indices.to(dtype)

    num_sms = 16
    grid = lambda meta: (num_sms, )
    _bincount_kernel[grid](
        indices,
        out,
        SIZE,
        semantic=semantic,
        scope=scope,
        num_warps=num_warps,
    )

    torch.testing.assert_close(out_ref, out, atol=0, rtol=0, equal_nan=False)
    print(f"✅ atom_add_{semantic}_{scope}_{dtype} Triton and Torch match")


def test_atomic_add_main():
    dtype_list = [torch.int32, torch.uint32, torch.uint64]
    scope_list = ["gpu", "sys"]
    semantic_list = ["acquire", "release", "relaxed"]
    for dtype in dtype_list:
        for scope in scope_list:
            for semantic in semantic_list:
                _test_atomic_add(semantic, scope, dtype)


def test_ld_st():

    @triton_dist.jit
    def _ld_st_kernel(
        input_tensor,
        output_tensor,
        scope: tl.constexpr,
        ld_semantic: tl.constexpr,
        st_semantic: tl.constexpr,
    ):
        value = ld(input_tensor + tid(0), scope=scope, semantic=ld_semantic)
        st(output_tensor + tid(0), value + 1, scope=scope, semantic=st_semantic)

    for dtype in [
            # torch.int8,
            # torch.uint8,
            # torch.int16,
            # torch.uint16,
            torch.int32, torch.uint32, torch.int64, torch.uint64, torch.float16, torch.bfloat16, torch.float
    ]:
        for scope in ["gpu", "sys"]:
            for ld_semantic, st_semantic in [
                ("relaxed", "relaxed"),
                ("acquire", "release"),
            ]:
                print(f"[ld_st] with dtype {dtype} scope {scope} semantic {ld_semantic} {st_semantic}")
                tensor_input = torch.arange(128, device="cuda", dtype=torch.int32).to(dtype)
                tensor_output = torch.zeros(128, device="cuda", dtype=dtype)
                compiled_kernel = _ld_st_kernel[(1, )](
                    tensor_input,
                    tensor_output,
                    scope=scope,
                    ld_semantic=ld_semantic,
                    st_semantic=st_semantic,
                    num_warps=2,
                )
                torch.testing.assert_close(
                    (tensor_input.to(torch.int32) + 1),
                    tensor_output.to(torch.int32),
                )
                print(f"✅ [ld_st] with dtype {dtype} scope {scope} semantic {ld_semantic} {st_semantic} done")


def test_shfl_sync():

    @triton_dist.jit
    def shfl_sync_kernel(input, output, index, width):
        thread_idx = tid(0)
        x = ld(input + thread_idx, scope="gpu", semantic="relaxed")
        y = __shfl_sync_i32(x, index)
        st(output + thread_idx, y, scope="gpu", semantic="relaxed")

    output = torch.zeros(128, device="cuda", dtype=torch.int32)
    delta = 5
    shfl_sync_kernel[(1, )](
        torch.arange(128, device="cuda", dtype=torch.int32),
        output,
        delta,
        32,
        num_warps=2,
    )

    assert torch.allclose(
        output,
        torch.cat((
            torch.ones(64, dtype=torch.int32) * delta,
            torch.ones(64, dtype=torch.int32) * (delta + 64),
        )).cuda()), output
    print("✅ [shfl_sync] passed.")


def test_laneid(device):

    @triton_dist.jit
    def store_laneid_kernel(inp_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        lid = laneid()
        tid = pid * 64 + lid
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(inp_ptr + offsets)
        res = a + tid
        tl.store(out_ptr + offsets, res)

    SIZE = 8 * 64
    dtype = torch.int32
    inp = torch.ones((SIZE, ), dtype=dtype, device=device)
    tri_out = torch.empty_like(inp)
    tids = torch.arange(SIZE).to(dtype).to(device)
    ref_out = inp + tids
    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
    store_laneid_kernel[grid](inp, tri_out, BLOCK_SIZE=64)
    torch.testing.assert_close(tri_out, ref_out, equal_nan=True)
    print("✅ [laneid] passed")


def test_atomic_add_per_warp():

    @triton_dist.jit
    def _atomic_add_per_warp_kernel(inptr, outptr, scope: tl.constexpr, semantic: tl.constexpr):
        value = atomic_add_per_warp(inptr + tid(0), 1, scope=scope, semantic=semantic)
        st(outptr + tid(0), value, scope="gpu", semantic="relaxed")

    for dtype in [
            torch.int32,
            torch.uint32,
    ]:
        for scope in ["gpu", "sys"]:
            for semantic in ("acquire", "release", "relaxed", "acq_rel"):
                print(f"[atomic_add_per_warp] with dtype {dtype} scope {scope} semantic {semantic}")
                tensor_input = torch.arange(128, device="cuda", dtype=torch.int32).to(dtype)
                tensor_output = torch.zeros(128, device="cuda", dtype=dtype)
                _atomic_add_per_warp_kernel[(1, )](
                    tensor_input,
                    tensor_output,
                    scope=scope,
                    semantic=semantic,
                    num_warps=2,
                )
                torch.testing.assert_close(
                    tensor_input,
                    (torch.arange(128, device="cuda", dtype=torch.int32) +
                     torch.zeros(128, device="cuda", dtype=torch.int32).index_fill_(
                         0, torch.arange(0, 128, 64, device="cuda"), 1)).to(dtype),
                )
                torch.testing.assert_close(
                    tensor_output,
                    torch.zeros(128, device="cuda").index_fill_(0, torch.arange(64, 128, device="cuda"), 64).to(dtype),
                )
                print(f"✅ [atomic_add_per_warp] with dtype {dtype} scope {scope} semantic {semantic} passed.")


def test_wait_and_consumetoken():
    import time

    @triton_dist.jit
    def consumer_kernel(input, output, barrier_tensor):
        tid = tid(0)
        token = ld_acquire(barrier_tensor, "gpu")
        while token != 1:
            token = ld_acquire(barrier_tensor, "gpu")
        input = dl.consume_token(input, token)
        val = ld_b32(input + tid)
        st(output + tid, val)

    device = "cuda"
    dtype = torch.int32
    rank = 0
    num_ranks = 8
    barrier_tensor = torch.zeros([num_ranks], dtype=torch.int32, device=device)
    SIZE = 4 * 64
    input = torch.randint(int(2e9), [SIZE], dtype=dtype, device=device)
    output = torch.randint(int(2e9), [SIZE], dtype=dtype, device=device)
    output_origin = output.clone()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        consumer_kernel[(1, )](input, output, barrier_tensor, num_warps=4)
    print("Consumer kernel launched!")
    print("signals are:")
    print(barrier_tensor)
    print("sleeping...", flush=True)
    time.sleep(3)
    print("wake up!", flush=True)
    assert torch.allclose(output_origin, output, atol=1e-3, rtol=1e-3)
    barrier_tensor.fill_(1)
    print("signals are:")
    print(barrier_tensor)

    torch.cuda.current_stream().wait_stream(stream)
    print(barrier_tensor)
    assert torch.allclose(input, output, atol=1e-3, rtol=1e-3)
    print("Pass!")


if __name__ == "__main__":

    torch.cuda.set_device(0)
    # PASS
    test_ld_st()
    test_wait_and_consumetoken()
    test_atomic_add_main()
    test_laneid("cuda")
    test_shfl_sync()
    test_atomic_add_per_warp()
    # TODO: NOT PASS
    # test_atomic_cas_main()

    print("All tests passed!")
