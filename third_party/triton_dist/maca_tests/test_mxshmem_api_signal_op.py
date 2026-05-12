import os
import datetime
import torch
import triton_dist
from triton_dist.language.extra.maca import libmxshmem_device as libshmem_device
from triton_dist.language.extra.maca.language_extra import tid, __syncthreads
import triton.pymxshmem as pymxshmem


@triton_dist.jit
def pingpong(t, iters):
    # pingpong for rank 0-1, 2-3, ...
    mype = libshmem_device.my_pe()
    thread_idx = tid(axis=0)
    if thread_idx == 0:
        for n in range(iters):
            if mype == 0:
                libshmem_device.signal_wait_until(t, libshmem_device.MXSHMEM_CMP_EQ, 1 + n)
                libshmem_device.signal_op(
                    t,
                    1 + n,
                    libshmem_device.MXSHMEM_SIGNAL_SET,
                    1,
                )
            elif mype == 1:
                libshmem_device.signal_op(
                    t,
                    1 + n,
                    libshmem_device.MXSHMEM_SIGNAL_SET,
                    0,
                )
                libshmem_device.signal_wait_until(t, libshmem_device.MXSHMEM_CMP_EQ, 1 + n)
    __syncthreads()


def test_pingpong():
    t = pymxshmem.mxshmem_create_tensor((1, ), torch.uint64)
    t.fill_(0)
    pymxshmem.mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    pingpong[(1, )](t, 100, num_warps=1)
    pymxshmem.mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    if RANK == 0:
        try:
            torch.testing.assert_close(t.to(torch.int32), torch.ones([1], dtype=torch.int32, device="cuda") * 100)
        except Exception as e:
            print("mxshmemx_signal with pingpong failed")
            raise e
        else:
            print("mxshmemx_signal with pingpong pass")


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    print("rank=", RANK, "world_size=", WORLD_SIZE)
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

    print("test mxshmemx_signal with pingpong...")
    test_pingpong()
    pymxshmem.mxshmem_finalize()

    torch.distributed.destroy_process_group()
