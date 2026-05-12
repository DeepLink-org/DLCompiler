import torch
from triton.language import core
import triton
import triton.language as tl
import triton_dist
from triton_dist.language.extra import language_extra
from triton_dist.language.extra.maca.language_extra import tid, st, ld
from triton_dist.language.extra.maca.memory_ops import (zero_vec_f32, unpack_bf16x2_f32, pack_f32_bf16x2, load_v4,
                                                        store_v4)


def test_set_zero():

    @triton_dist.jit
    def set_zero_kernel(inp_ptr, out_ptr, vec: tl.constexpr):
        tid = tid(0)
        offsets = tid * vec
        if vec == 1:
            a = ld(inp_ptr + offsets)
            res = zero_vec_f32(1)
            res += a
            st(out_ptr + offsets, tl.cast(res, tl.int32))
        elif vec == 2:
            a0 = ld(inp_ptr + offsets)
            a1 = ld(inp_ptr + offsets + 1)
            res0, res1 = zero_vec_f32(2)
            res0 += a0
            res1 += a1
            st(out_ptr + offsets, tl.cast(res0, tl.int32))
            st(out_ptr + offsets + 1, tl.cast(res1, tl.int32))

    for VEC_SIZE in [1, 2]:
        SIZE = VEC_SIZE * 64
        dtype = torch.int32
        inp = torch.empty((SIZE, ), dtype=dtype, device="cuda")
        inp.fill_(123)
        out = torch.ones((SIZE, ), dtype=dtype, device="cuda")
        # ref_out = torch.zeros((SIZE, ), dtype=dtype, device="cuda") + inp
        set_zero_kernel[(1, )](inp, out, VEC_SIZE, num_warps=1)
        torch.testing.assert_close(out, inp)
    print("✅ [set zero] passed")


def test_ld_st_v4(dtype):

    device = torch.cuda.current_device()
    # VEC_SIZE = 16/dtype.itemsize    # 16B / item_byte_width

    @triton_dist.jit
    def _ld_st_kernel(
        input,
        out,
        size,
    ):
        threads_per_block = language_extra.num_threads()
        tid = tid(0)
        pid = tl.program_id(axis=0)
        num_pid = tl.num_programs(axis=0)
        total_threads = threads_per_block * num_pid
        global_tid = pid * threads_per_block + tid

        v0, v1, v2, v3 = load_v4(input + global_tid * 4, "b32")
        store_v4(out + global_tid * 4, v0, v1, v2, v3, "b32")

    SIZE = 2 * 64 * 4
    x = torch.randint(int(2e9), (SIZE, ), dtype=torch.int32, device=device).to(dtype)
    z = torch.empty((SIZE, ), dtype=dtype, device=device)
    num_warps = 2
    grid = lambda meta: (1, )
    _ld_st_kernel[grid](
        x,
        z,
        SIZE,
        num_warps=num_warps,
    )
    torch.testing.assert_close(x, z, atol=0, rtol=0, equal_nan=False)
    print("✅ [ld_st_v4] passed")


@triton_dist.jit
def _test_ld_unpack_pack_st_kernel(input_ptr, output_ptr, rows, cols, VEC_SIZE: tl.constexpr):

    threads_per_block = language_extra.num_threads()
    tid = tid(0)
    pid = tl.program_id(axis=0)
    global_tid = pid * threads_per_block + tid
    tid_offset = global_tid * VEC_SIZE

    acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8 = zero_vec_f32(VEC_SIZE)

    for i in range(rows):
        offset = i * cols + tid_offset
        t1, t2, t3, t4 = load_v4(input_ptr + offset, "b32")
        u1, u2, u3, u4, u5, u6, u7, u8 = unpack_bf16x2_f32(t1, t2, t3, t4)  # t1:bf16x2 -> u1,u2:f32
        acc1 += u1
        acc2 += u2
        acc3 += u3
        acc4 += u4
        acc5 += u5
        acc6 += u6
        acc7 += u7
        acc8 += u8

    v1, v2, v3, v4 = pack_f32_bf16x2(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)  #acc1,acc2:f32 -> v1:bf16x2
    store_v4(output_ptr + tid_offset, v1, v2, v3, v4, "b32")


def test_ld_unpack_pack_st():
    device = torch.cuda.current_device()
    VEC_SIZE = 8
    ROWS = 16
    COLS = 4096
    x = torch.randn((ROWS, COLS), dtype=torch.bfloat16, device=device)
    z = torch.empty(COLS, dtype=torch.bfloat16, device=device)
    num_warps = 2
    grid = (triton.cdiv(COLS, num_warps * 64 * VEC_SIZE), )
    _test_ld_unpack_pack_st_kernel[grid](x, z, ROWS, COLS, VEC_SIZE)
    ref = x.to(torch.float).sum(dim=0).to(torch.bfloat16)
    torch.testing.assert_close(ref, z, atol=0, rtol=0, equal_nan=False)
    print("✅ [ld_unpack_pack_st] passed")


if __name__ == "__main__":

    torch.cuda.set_device(0)
    test_set_zero()
    test_ld_st_v4(torch.float)
    test_ld_unpack_pack_st()
    print("All tests passed!")
