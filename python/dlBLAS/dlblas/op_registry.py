import os
from dataclasses import dataclass, field
from typing import Optional

from triton.runtime.jit import JITFunction

from dlblas.op_struct import OpImpl, OpParams, parse_args, match
from dlblas.cache import Cache
from dlblas.autotune.space import ChoiceSpace, DictSpace
from dlblas.dynamic_compiler import compile_op, tunning


@dataclass
class OpRegistry:
    # ops: name -> list[OpImpl]
    ops: dict[str, OpImpl] = field(default_factory=dict)
    cache: Cache = field(default_factory=Cache)

    def __post_init__(self):
        # XXX? To use CUDA with multiprocessing, you must use the 'spawn' start method?
        # import multiprocessing
        # multiprocessing.set_start_method('spawn')

        # TODO also read cache from file?
        pass

    def register(self, name, spaces, args, call, bench_fn, kernel):
        assert isinstance(
            kernel,
            JITFunction), f'kernel must be JITFunction, but got {type(kernel)}'
        assert isinstance(
            spaces, (ChoiceSpace, DictSpace)
        ), f'space must be ChoiceSpace or DictSpace, but got {type(spaces)}'
        params = parse_args(args)

        # path-to-deeplink/python/dlBLAS/dlblas
        this_file_dir = os.path.dirname(os.path.realpath(__file__))
        # kernel_file_name = call.__globals__['__name__']
        kernel_module_name = call.__module__
        kernel_file = os.path.join(this_file_dir, 'kernels',
                                   kernel_module_name + '.py')
        impl = OpImpl(
            params,
            kernel_file,
            None,
            spaces,
            call,
            bench_fn,
            kernel,
        )

        # FIXME what if a kernel register twice? if appear seems to be a bug... de-duplication check
        if name in self.ops:
            self.ops[name].append(impl)
        else:
            self.ops[name] = [impl]

    def get_list_op_names(self):
        return list(self.ops.keys())

    def get_args_from_op_name(self, op_name: str):
        return [i.params for i in self.ops[op_name]]

    def get_op(self, op_name: str, args: tuple):
        if op_name not in self.ops:
            raise NameError(f"op {op_name} not found")

        # 1. check cache
        if op := self.look_up_cache(op_name, args):
            # if op is not None, will hit the true branch
            return op

        # 2. if miss, tunning
        op = self._tunning(op_name, args)
        return op

    def look_up_cache(self, op_name: str, args: tuple) -> Optional[OpImpl]:
        if cached := self.cache.get(op_name, args):
            assert isinstance(cached, OpImpl)
            compile_op(cached)
            return cached

    def _tunning(self, op_name: str, args: tuple):
        # fetch candidates
        candidates = self._get_candidates(op_name, args)
        if len(candidates) == 0:
            raise LookupError(
                f"no candidates for op {op_name} with args {args}")

        # run selection
        best_idx, _ = self._selection(args, candidates)

        # get best
        best_op: OpImpl = candidates[best_idx]

        # cache
        self.cache.put(best_op, op_name, args)
        return best_op

    def _get_candidates(self, op_name: str, args: tuple):
        candidates = []
        for op in self.ops[op_name]:
            # XXX the same op can have multiple dtype, impl, device etc
            # we might want to shorten look up time, by
            # hash those info when registering op
            if match(args, op.params):
                candidates.append(op)
        return candidates

    def _selection(self, args, candidates: list[OpImpl]) -> int:
        # NOTE: for now we only bench each one locally and in serial
        # for parallel benchmark, see:
        # https://github.com/pytorch/pytorch/blob/a0dac3de31b50a19e503652ffe257b314fa005a6/torch/_inductor/autotune_process.py#L282
        best_idx = -1
        best_perf = None
        for i, op in enumerate(candidates):
            # perf = op.bench(*args)
            perf = tunning(op, args)

            # print('op is : ', op, ' perf is: ', perf)
            if best_perf is None or perf < best_perf:
                best_perf = perf
                best_idx = i
        return best_idx, best_perf

    # def _bench(self, op: OpImpl):
    #     # open a subprocess and run the benchmark

    #     ## NOTE: the driver code must wrap within if __name__ == '__main__'
    #     ## To use CUDA with multiprocessing, you must use the 'spawn' start method
    #     mp_context = multiprocessing.get_context('spawn')

    #     queue = mp_context.Queue()
    #     # https://pytorch.org/docs/stable/notes/multiprocessing.html
    #     # When a Tensor is sent to another process, the Tensor data is shared.
    #     process = mp_context.Process(
    #         target=op.kernel,
    #         args=(
    #             # mp
    #             queue,
    #         ))
    #     process.start()
    #     process.join()
    #     perf = queue.get()

    #     # return the performance
    #     return perf


op_registry = OpRegistry()
