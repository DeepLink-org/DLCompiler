import os
import time
import datetime
import triton
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


def test_mxshmem_ptr():
    N = 1
    dtype = torch.int32

    @triton_dist.jit
    def _mxshmem_st(ptr, num_ranks: tl.constexpr):
        for n in range(num_ranks):
            mc_ptr = libshmem_device.remote_ptr(ptr, n).to(tl.pointer_type(tl.int32))
            data = tl.load(ptr)
            tl.store(mc_ptr, data)

    t: torch.Tensor = pymxshmem.mxshmem_create_tensor((N, ), dtype)
    t.fill_(1 + RANK)
    pymxshmem.mxshmem_barrier_all()
    if RANK == 0:
        _mxshmem_st[(N, )](t, num_ranks=WORLD_SIZE)
    pymxshmem.mxshmem_barrier_all()
    print("123")
    torch.testing.assert_close(t, torch.ones_like(t))
    print("done")


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

    test_mxshmem_ptr()

    torch.distributed.barrier(TP_GROUP)
    pymxshmem.mxshmem_finalize()
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()
