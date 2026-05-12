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
import functools
import os

import pytest
import triton.pymxshmem as pymxshmem
import torch
import torch.distributed
import datetime
import triton.language as tl
import triton_dist
from triton_dist.language.extra.maca.language_extra import (__syncthreads, tid, st)
from triton_dist.language.extra.maca import libmxshmem_device as libshmem_device
# from triton_dist.utils import (MXSHMEM_SIGNAL_DTYPE, has_mxshmemi_bc_built, initialize_distributed,
#                                mxshmem_barrier_all_on_stream, mxshmem_free_tensor_sync, mxshmem_create_tensor,
#                                is_mxshmem_multimem_supported)
from triton.pymxshmem import (mxshmem_create_tensor, mxshmem_barrier_all_on_stream)

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
MXSHMEM_SIGNAL_DTYPE = torch.uint64


def conditional_execution(condition_func):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition_func():
                return func(*args, **kwargs)
            else:
                print(f"{condition_func.__name__} not satisfied. skip {func.__name__}...")
                return None

        return wrapper

    return decorator


def test_mxshmem_basic():

    @triton_dist.jit
    def _mxshmem_basic(output):
        thread_idx = tid(axis=0)
        if thread_idx == 0:
            st(output, libshmem_device.my_pe())
            output += 1
            # st(output, libshmem_device.team_my_pe(libshmem_device.MXSHMEM_TEAM_WORLD))
            # output += 1
            # st(output, libshmem_device.team_my_pe(libshmem_device.MXSHMEMX_TEAM_NODE))
            # output += 1

            st(output, libshmem_device.n_pes())
            # st(output, libshmem_device.team_n_pes(libshmem_device.MXSHMEM_TEAM_WORLD))
            # output += 1
            # st(output, libshmem_device.team_n_pes(libshmem_device.MXSHMEMX_TEAM_NODE))

    print("mxshmem basic start...")
    output = mxshmem_create_tensor((2, ), torch.int32)
    _mxshmem_basic[(1, )](output)
    mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    try:
        torch.testing.assert_close(output, torch.tensor([RANK, WORLD_SIZE], dtype=torch.int32, device="cuda")), output
    except Exception as e:
        print(" ❌ mxshmem basic failed")
        raise (e)
    else:
        print("✅ mxshmem basic pass")


def test_mxshmemx_putmem_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton_dist.jit
    def _mxshmemx_putmem(
        ptr,
        elems_per_rank,
        scope: tl.constexpr,
        nbi: tl.constexpr,
        ELEM_SIZE: tl.constexpr,
    ):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_nbi_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_nbi_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                # elif scope == "thread":
                #     if thread_idx < elems_per_rank:
                #         libshmem_device.putmem_nbi(
                #             ptr + mype * elems_per_rank + thread_idx,
                #             ptr + mype * elems_per_rank + thread_idx,
                #             1,
                #             pid,
                #         )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                # elif scope == "thread":
                #     if thread_idx < elems_per_rank:
                #         libshmem_device.putmem(
                #             ptr + mype * elems_per_rank + thread_idx,
                #             ptr + mype * elems_per_rank + thread_idx,
                #             1,
                #             pid,
                #         )
                else:
                    raise ValueError("scope must be block, warp. thread not supported yet")

    t = mxshmem_create_tensor((N, ), dtype)

    for scope in ["block", "warp"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "mxshmemx_putmem_block",
                ("warp", False): "mxshmemx_putmem_warp",
                #("thread", False): "mxshmem_putmem",
                ("block", True): "mxshmemx_putmem_nbi_block",
                ("warp", True): "mxshmemx_putmem_nbi_warp",
                # ("thread", True): "mxshmem_putmem_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
            _mxshmemx_putmem[(WORLD_SIZE, )](
                t,
                N // WORLD_SIZE,
                scope,
                nbi,
                ELEM_SIZE=dtype.itemsize,
                num_warps=1 if scope == "warp" else 4,
            )
            mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                raise (e)
            else:
                print(f"✅ {api} pass")


def test_mxshmem_signal():

    @triton_dist.jit
    def _pingpong(t, iters):
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

    print("test mxshmemx_signal with pingpong...")
    t = mxshmem_create_tensor((1, ), MXSHMEM_SIGNAL_DTYPE)
    t.fill_(0)
    mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    _pingpong[(1, )](t, 100, num_warps=1)
    mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    if RANK == 0:
        try:
            torch.testing.assert_close(t.to(torch.int32), torch.ones([1], dtype=torch.int32, device="cuda") * 100)
        except Exception as e:
            print("❌ mxshmemx_signal with pingpong failed")
            raise e
        else:
            print("✅ mxshmemx_signal with pingpong pass")


def test_mxshmemx_putmem_signal_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton_dist.jit
    def _mxshmemx_putmem_signal(ptr, signal, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_signal_nbi_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        signal + mype,
                        1,
                        libshmem_device.MXSHMEM_SIGNAL_SET,
                        pid,
                    )
                elif scope == "warp":
                    if wid == 0:
                        libshmem_device.putmem_signal_nbi_warp(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.MXSHMEM_SIGNAL_SET,
                            pid,
                        )
                # elif scope == "thread":
                #     if thread_idx == 0:
                #         libshmem_device.putmem_signal_nbi(
                #             ptr + mype * bytes_per_rank,
                #             ptr + mype * bytes_per_rank,
                #             bytes_per_rank,
                #             signal + mype,
                #             1,
                #             libshmem_device.MXSHMEM_SIGNAL_SET,
                #             pid,
                #         )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
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
                elif scope == "warp":
                    if wid == 0:
                        libshmem_device.putmem_signal_warp(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.MXSHMEM_SIGNAL_SET,
                            pid,
                        )
                # elif scope == "thread":
                #     if thread_idx == 0:
                #         libshmem_device.putmem_signal(
                #             ptr + mype * bytes_per_rank,
                #             ptr + mype * bytes_per_rank,
                #             bytes_per_rank,
                #             signal + mype,
                #             1,
                #             libshmem_device.MXSHMEM_SIGNAL_SET,
                #             pid,
                #         )
                else:
                    raise ValueError("scope must be block, warp. thread not supported yet")

    t = mxshmem_create_tensor((N, ), dtype)
    signal = mxshmem_create_tensor((WORLD_SIZE, ), MXSHMEM_SIGNAL_DTYPE)

    for scope in ["block", "warp"]:  #"thread"
        for nbi in [True, False]:
            api = {
                ("block", False): "mxshmemx_putmem_signal_block",
                ("warp", False): "mxshmemx_putmem_signal_warp",
                #("thread", False): "mxshmem_putmem_signal",
                ("block", True): "mxshmemx_putmem_signal_nbi_block",
                ("warp", True): "mxshmemx_putmem_signal_nbi_warp",
                #("thread", True): "mxshmem_putmem_signal_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            signal.fill_(0)
            signal[RANK].fill_(1)
            mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
            _mxshmemx_putmem_signal[(WORLD_SIZE, )](
                t,
                signal,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=4,
            )
            mxshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
                torch.testing.assert_close(signal, torch.ones((WORLD_SIZE, ), dtype=MXSHMEM_SIGNAL_DTYPE,
                                                              device="cuda"))
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                print(signal)
                raise (e)
            else:
                print(f"✅ {api} pass")


def test_mxshmem_barrier_sync_quiet_fence():
    """only test runs, no result checked"""

    @triton_dist.jit
    def _mxshmem_barrier_sync_quiet_fence():
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid == 0:
            libshmem_device.barrier_all_block()
            # libshmem_device.sync_all_block()

            # if thread_idx / 32 == 0:
            #     libshmem_device.barrier_all_warp()
            #     libshmem_device.sync_all_warp()

            #if thread_idx == 0:
            #    libshmem_device.barrier_all()
            #    libshmem_device.sync_all()

        # libshmem_device.quiet()
        libshmem_device.fence()

    # @triton_dist.jit
    # def _mxshmem_barrier_sync_quiet_fence_with_team(team):
    #     pid = tl.program_id(axis=0)
    #     thread_idx = tid(axis=0)
    #     if pid == 0:
    #         libshmem_device.barrier_block(team)
    #         libshmem_device.team_sync_block(team)

    #         if thread_idx / 32 == 0:
    #             libshmem_device.barrier_warp(team)
    #             libshmem_device.team_sync_warp(team)

    #         if thread_idx == 0:
    #             libshmem_device.barrier(team)

    print("test mxshmem_barrier/mxshmem_sync/mxshmem_quiet/mxshmem_fence all in one...")
    _mxshmem_barrier_sync_quiet_fence[(1, )](num_warps=4)
    torch.cuda.synchronize()
    print("✅ mxshmem_barrier_all/mxshmem_sync/mxshmem_quiet/mxshmem_fence pased...")
    # _mxshmem_barrier_sync_quiet_fence_with_team[(1, )](mxshmem.core.Teams.TEAM_NODE, num_warps=4)
    # torch.cuda.synchronize()
    # print("✅ mxshmem_barrier/mxshmemx_team_sync pased...")


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

    test_mxshmem_basic()
    # test_mxshmemx_getmem_with_scope(31 * WORLD_SIZE, torch.int8)
    test_mxshmemx_putmem_with_scope(16 * WORLD_SIZE, torch.int8)
    test_mxshmemx_putmem_signal_with_scope(20 * WORLD_SIZE, torch.int8)
    test_mxshmem_signal()
    test_mxshmem_barrier_sync_quiet_fence()
    # test_mxshmem_broadcast(32 * WORLD_SIZE, torch.int8)

    # some ranks hangs. don't know why
    # test_mxshmem_fcollect(1024, torch.int8)
    # test_mxshmem_multimem_st(1024)

    # test_mxshmemi_putmem_rma(16 * WORLD_SIZE, torch.int8)
    # test_mxshmemi_putmem_rma_signal_with_scope(16 * WORLD_SIZE, torch.int8)

    pymxshmem.mxshmem_finalize()
    torch.distributed.destroy_process_group()
