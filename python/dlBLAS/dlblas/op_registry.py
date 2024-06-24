from dataclasses import dataclass, field, astuple
from typing import Any


@dataclass(eq=False)
class OpParams:
    n_args: int
    args_names: list

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
        if len(self.shapes) != len(other.shapes):
            return False
        if len(self.dtypes) != len(other.dtypes):
            return False
        if len(self.device) != len(other.device):
            return False

        for i in range(self.n_args):
            if self.args_names[i] != other.args_names[i]:
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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.kernel(*args, **kwds)


@dataclass
class OpRegistry:
    # ops: name -> list[OpImpl]
    ops: dict[str, OpImpl] = field(default_factory=lambda: {})

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

        best_idx = -1
        best_perf = None
        for i, op in enumerate(candidates):
            perf = self._bench(op)
            if best_perf is None or perf > best_perf:
                best_perf = perf
                best_idx = i

        return best_idx, best_perf

    def _bench(self, op: OpImpl):
        # TODO

        # prepare args

        # open a subprocess and run the benchmark

        # return the performance
        return 0


op_registry = OpRegistry()
