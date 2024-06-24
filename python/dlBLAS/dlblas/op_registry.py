import torch

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpParams:
    shapes: list
    rank: int
    dtypes: list = field(default_factory=lambda: [torch.float32])


@dataclass
class OpImpl:
    params: OpParams
    kernel: callable

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.kernel(self.params, *args, **kwds)


# TODO figoure out a way to automatically import all files from kernels/


@dataclass
class OpRegistry:
    # ops: name -> list[OpImpl]
    ops: dict = field(default_factory=lambda: {})

    def register(self, name, impl: OpImpl):
        if name in self.ops:
            self.ops[name].append(impl)
        else:
            self.ops[name] = [impl]


op_registry = OpRegistry()
