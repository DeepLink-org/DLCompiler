import torch
import os
import datetime
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton.pymaca.maca as maca


def test_batchcpy(rank, num_ranks, pg):

    flush_cache = torch.rand(1024 * 1024 * 64, dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream

    dst = symm_mem.empty(256, dtype=torch.float32, device=f"cuda:{rank}")
    dst.fill_(rank + 1)
    symm_mem_hdl = symm_mem.rendezvous(dst, group=dist.group.WORLD)
    assert symm_mem_hdl is not None, "a_shard must be allocated via SymmetricMemory"

    src = torch.empty(256, device=f"cuda:{rank}", dtype=torch.float32)
    src.fill_(rank + 1)
    num_chunk = 4
    ele_byte = torch.finfo(torch.float32).bits // 8
    chunk_size = 256 // num_chunk
    # actually barrier should be uncached
    barrier_tensor = symm_mem.empty(num_chunk, device=f"cuda:{rank}", dtype=torch.uint64)
    barrier_hdl = symm_mem.rendezvous(barrier_tensor, group=dist.group.WORLD)
    barrier_value = 1

    dst_arr = []
    src_arr = []
    engine = []
    count = []
    write_flag = []
    write_value = []
    dst_buf = symm_mem_hdl.get_buffer((rank + 1) % num_ranks, (256, ), torch.float32, 0)
    barrier_buf = barrier_hdl.get_buffer((rank + 1) % num_ranks, (num_chunk, ), torch.uint64, 0)

    for i in range(num_chunk):
        dst_arr.append(dst_buf[i * chunk_size].data_ptr())
        src_arr.append(src[i * chunk_size].data_ptr())
        engine.append(maca.mcParallelCopyEngine.ParallelCopyEngineDefault)
        count.append(chunk_size * ele_byte)
        write_flag.append(barrier_buf[i].data_ptr())
        write_value.append(barrier_value)

    flush_cache.fill_(1.0)
    (err, ) = maca.mcExtBatchCopyFlagAndWait(dst_arr, src_arr, engine, count, write_flag, write_value, [], [], stream)

    torch.cuda.synchronize()

    sent_rank = (rank - 1 + num_ranks) % num_ranks
    ref_dst = torch.empty(256, device=f"cuda:{rank}", dtype=torch.float32)
    ref_barrier = symm_mem.empty(num_chunk, device=f"cuda:{rank}", dtype=torch.uint64)
    ref_dst.fill_(sent_rank + 1)
    ref_barrier.fill_(1)

    try:
        torch.testing.assert_close(dst, ref_dst)
        torch.testing.assert_close(barrier_tensor, ref_barrier)
    except Exception as e:
        print(f"ERROR:mcExtBatchCopyFlagAndWait failed")
        print(dst)
        print(barrier_tensor)
        raise (e)
    else:
        print(f"mcExtBatchCopyFlagAndWait pass")


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)

    assert LOCAL_RANK == RANK

    test_batchcpy(RANK, WORLD_SIZE, TP_GROUP)

    torch.distributed.destroy_process_group()
