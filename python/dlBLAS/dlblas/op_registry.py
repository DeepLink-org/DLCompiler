from dataclasses import dataclass, field, astuple
from typing import Any

import multiprocessing

import numpy as np


@dataclass(eq=False)
class OpParams:
    n_args: int
    args_names: list
    args_types: list

    shapes: dict  # args_name -> shape
    dtypes: dict  # args_name -> dtype
    device: dict  # args_name -> device

    def __eq__(self, other):
        if not isinstance(other, OpParams):
            return False
        if self.n_args != other.n_args:
            return False
        if len(self.args_names) != len(other.args_names):
            return False
        if len(self.args_types) != len(other.args_types):
            return False
        if len(self.shapes) != len(other.shapes):
            return False
        if len(self.dtypes) != len(other.dtypes):
            return False
        if len(self.device) != len(other.device):
            return False

        for i in range(self.n_args):
            if self.args_names[i] != other.args_names[i]:
                return False
            if self.args_types[i] != other.args_types[i]:
                # FIXME type comp?
                return False
            if self.shapes[self.args_names[i]] != other.shapes[
                    self.args_names[i]]:
                return False
            if self.dtypes[self.args_names[i]] != other.dtypes[
                    self.args_names[i]]:
                return False
            if self.device[self.args_names[i]] != other.device[
                    self.args_names[i]]:
                return False

        return True


@dataclass
class OpImpl:
    params: OpParams
    kernel: callable
    bench_fn: callable

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.kernel(*args, **kwds)


@dataclass
class OpRegistry:
    # ops: name -> list[OpImpl]
    ops: dict[str, OpImpl] = field(default_factory=lambda: {})

    def __post_init__(self):
        # XXX? To use CUDA with multiprocessing, you must use the 'spawn' start method
        multiprocessing.set_start_method('spawn')

    def register(self, name, impl: OpImpl):
        if name in self.ops:
            self.ops[name].append(impl)
        else:
            self.ops[name] = [impl]

    def get_list_op_names(self):
        return list(self.ops.keys())

    def get_args_from_op_name(self, op_name: str):
        return [i.params for i in self.ops[op_name]]

    def get_op(self, op_name: str, params: OpParams):
        if op_name not in self.ops:
            raise NameError(f"op {op_name} not found")

        # fetch candidates
        candidates = []
        for op in self.ops[op_name]:
            if op.params == params:
                candidates.append(op)

        if len(candidates) == 0:
            raise LookupError(
                f"no candidates for op {op_name} with params {astuple(params)}"
            )

        # run selection
        best_idx, _ = self._selection(candidates)

        # get best
        best_op = candidates[best_idx]
        return best_op

    def _selection(self, candidates: list[OpImpl]) -> int:
        if len(candidates) == 1:
            return 0, 0

        # NOTE: Even though we want tasks to run one at a time, we still use Pool
        # Setting processes to 1 ensures serial execution
        futures = []
        with multiprocessing.Pool(processes=1) as pool:
            for i, op in enumerate(candidates):
                # TODO construct params
                args = ()
                future = pool.apply(op.bench_fn, args=args)
                futures.append(future)

            results = [future.get() for future in futures]

        index = np.argmax(results)
        return int(index), max(results)

    def _build_args(self, params: OpParams):
        # need to figure out symbolic shape 'n'
        # and concrete shape 16 etc...
        pass

    # def _selection(self, candidates: list[OpImpl]) -> int:
    #     if len(candidates) == 1:
    #         return 0, 0

    #     best_idx = -1
    #     best_perf = None
    #     for i, op in enumerate(candidates):
    #         perf = self._bench(op)
    #         if best_perf is None or perf > best_perf:
    #             best_perf = perf
    #             best_idx = i

    #     return best_idx, best_perf

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
