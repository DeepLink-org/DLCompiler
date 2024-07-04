import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union, Optional
import itertools

from dlblas.autotune.space import ChoiceSpace, DictSpace, RangeSapce, DiscreteSpace, PowerOfTwoSpace, FixedSpace
from dlblas.autotune.configs import AutotuneConfig


@dataclass
class Policy:
    space: Union[ChoiceSpace, DictSpace]

    def __post_init__(self):
        if isinstance(self.space, ChoiceSpace):
            pass
        elif isinstance(self.space, DictSpace):
            pass
        else:
            raise TypeError(
                f"space must be ChoiceSpace or DictSpace, but got {type(self.space)}"
            )

    def generate(self) -> Optional[dict]:
        raise NotImplementedError()

    def feedback(self, perf):
        pass


@dataclass
class RandomPolicy(Policy):

    def generate(self):
        return self.space.sample()


@dataclass
class EnumerationPolicy(Policy):
    args_names: list[str] = field(default_factory=list)
    iters: Iterable = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.space, ChoiceSpace):
            self.iters = iter(self.space.to_iter())

        elif isinstance(self.space, DictSpace):
            args_names = []
            iters = []
            for name, subspace in self.space.params.items():
                args_names.append(name)
                if isinstance(subspace, RangeSapce):
                    raise RuntimeError(
                        "EnumerationPolicy doesn't support RangeSpace")
                elif isinstance(subspace, FixedSpace):
                    iters.append(subspace.to_iter())
                elif isinstance(subspace, DiscreteSpace):
                    iters.append(subspace.to_iter())
                elif isinstance(subspace, PowerOfTwoSpace):
                    iters.append(subspace.to_iter())
                else:
                    raise TypeError(f"unsupported type {type(subspace)}")

            iters = itertools.product(*iters)
            self.iters = iters
            self.args_names = args_names

    def generate(self):
        try:
            next_item = next(self.iters)
            if isinstance(self.space, ChoiceSpace):
                return next_item
            elif isinstance(self.space, DictSpace):
                ans = {}
                for k, v in zip(self.args_names, next_item):
                    ans[k] = v
                return ans

        except StopIteration:
            return None


def get_policy(spaces, configs: AutotuneConfig):
    if configs.tunner == 'random':
        return RandomPolicy(spaces)
    elif configs.tunner == 'enumeration':
        return EnumerationPolicy(spaces)
    raise NameError(f"tunner {configs.tunner} not found")
