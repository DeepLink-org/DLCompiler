import torch
import triton.language as tl
from triton_dist.utils import (
    MACA_CHECK, )
import triton.pymaca.maca as maca


@tl.core.extern
def thread_id(axis: tl.constexpr, _semantic=None):
    return tl.inline_intrinsic_elementwise(
        intrinsic=f"llvm.mxc.thread.id.{axis.value}",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        _semantic=_semantic,
    )


def _wait_eq_maca(ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        (err, ) = maca.mcStreamWaitValue32(
            stream.cuda_stream,
            ptr,
            signal,
            maca.mcStreamWaitValue_flags.MC_STREAM_WAIT_VALUE_EQ,
        )
    else:
        (err, ) = maca.mcStreamWaitValue64(
            stream.cuda_stream,
            ptr,
            signal,
            maca.mcStreamWaitValue_flags.MC_STREAM_WAIT_VALUE_EQ,
        )
    MACA_CHECK(err)


def _set_signal_maca(ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        (err, ) = maca.mcStreamWriteValue32(
            stream.cuda_stream,
            ptr,
            signal,
            maca.mcStreamWriteValue_flags.MC_STREAM_WRITE_VALUE_DEFAULT,
        )
    else:
        (err, ) = maca.mcStreamWriteValue64(
            stream.cuda_stream,
            ptr,
            signal,
            maca.mcStreamWriteValue_flags.MC_STREAM_WRITE_VALUE_DEFAULT,
        )
    MACA_CHECK(err)
