import os
import datetime
import torch
import triton_dist
from triton_dist.language.extra.maca import libmxshmem_device as libshmem_device
import triton.pymxshmem as pymxshmem
import triton.language as tl

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


@triton_dist.jit
def get_pe_and_npe():
    mype = libshmem_device.my_pe()
    npes = libshmem_device.n_pes()
    return (mype, npes)


@triton_dist.jit
def ring_put(ptr):
    mype, npes = get_pe_and_npe()
    peer = (mype + 1) % npes
    libshmem_device.int_p(ptr, mype, peer)


def test_ring_put():
    t = pymxshmem.mxshmem_create_tensor([32], torch.int)
    ring_put[(1, )](t)


# @triton.jit
# def peer(ptr):
#     mype = libshmem_device.my_pe()
#     npes = libshmem_device.n_pes()
#     peer = (mype + 1) % npes
#     libshmem_device.int_p(ptr, mype, peer)
#tl.store(ptr, peer)

if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.cuda.synchronize()
    pymxshmem.init_mxshmem_by_uniqueid(TP_GROUP)

    test_ring_put()

    pymxshmem.mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    pymxshmem.mxshmem_finalize()
    torch.cuda.synchronize()
    print("done")
    torch.distributed.destroy_process_group()
