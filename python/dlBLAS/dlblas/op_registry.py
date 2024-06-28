from dataclasses import dataclass, field

import multiprocessing

import numpy as np

from dlblas.op_struct import OpImpl, OpParams, parse_args, match


@dataclass
class OpRegistry:
    # ops: name -> list[OpImpl]
    ops: dict[str, OpImpl] = field(default_factory=lambda: {})

    def __post_init__(self):
        # XXX? To use CUDA with multiprocessing, you must use the 'spawn' start method
        multiprocessing.set_start_method('spawn')

    def register(self, name, args: tuple, call, bench_fn):
        params = parse_args(args)
        impl = OpImpl(params, call, bench_fn)

        # FIXME what if a kernel register twice? de-duplication check
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

        # fetch candidates
        candidates = []
        for op in self.ops[op_name]:
            # we don't expect too much time waste look up here,
            # we can always improve this look up time
            if match(args, op.params):
                candidates.append(op)

        if len(candidates) == 0:
            raise LookupError(
                f"no candidates for op {op_name} with args {args}")

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
