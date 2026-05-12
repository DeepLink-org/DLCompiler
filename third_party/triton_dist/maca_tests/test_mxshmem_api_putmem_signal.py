import os
import time
import datetime
import triton_dist
import triton.language as tl
from triton_dist.language.extra.maca import libmxshmem_device as libshmem_device
from triton_dist.language.extra.maca.language_extra import tid
import torch
import torch.distributed
import triton.pymxshmem as pymxshmem

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def test_mxshmemx_putmem_signal_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton_dist.jit
    def _mxshmemx_putmem_signal(ptr, signal, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if pid != mype:
            if scope == "block":
                libshmem_device.putmem_signal_block(
                    ptr + mype * bytes_per_rank,
                    ptr + mype * bytes_per_rank,
                    bytes_per_rank,
                    signal + mype,
                    1,
                    libshmem_device.MXSHMEM_SIGNAL_SET,
                    pid,
                )
            else:
                raise ValueError("scope must be block, warp, or thread")

    t = pymxshmem.mxshmem_create_tensor((N, ), dtype)
    signal = pymxshmem.mxshmem_create_tensor((WORLD_SIZE, ), torch.uint64)

    for scope in ["block"]:
        for nbi in [False]:
            api = {
                ("block", False): "mxshmemx_putmem_signal_block",
                ("warp", False): "mxshmemx_putmem_signal_warp",
                ("thread", False): "mxshmem_putmem_signal",
                ("block", True): "mxshmemx_putmem_signal_nbi_block",
                ("warp", True): "mxshmemx_putmem_signal_nbi_warp",
                ("thread", True): "mxshmem_putmem_signal_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            signal.fill_(0)
            signal[RANK].fill_(1)
            pymxshmem.mxshmem_barrier_all()
            _mxshmemx_putmem_signal[(WORLD_SIZE, )](
                t,
                signal,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=4,
            )
            pymxshmem.mxshmem_barrier_all()
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
                torch.testing.assert_close(signal, torch.ones((WORLD_SIZE, ), dtype=torch.uint64, device="cuda"))
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                print(signal)
                raise (e)
            else:
                print(f"✅ {api} pass")


if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.cuda.synchronize()
    pymxshmem.init_mxshmem_by_uniqueid(TP_GROUP)

    test_mxshmemx_putmem_signal_with_scope(20 * WORLD_SIZE, torch.int8)

    torch.distributed.barrier(TP_GROUP)
    torch.cuda.synchronize()
    pymxshmem.mxshmem_finalize()
    torch.distributed.destroy_process_group()
