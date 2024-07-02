import os
from dataclasses import dataclass, field

from dlblas.autotune.space import ChoiceSpace, DictSpace


@dataclass
class Policy:
    name: str

    def generate(self, space):
        raise NotImplementedError()


class RandomPolicy(Policy):

    def generate(self, space):
        if isinstance(space, ChoiceSpace):
            return space.sample()
        elif isinstance(space, DictSpace):
            return space.sample()
        else:
            raise TypeError(
                f"space must be ChoiceSpace or DictSpace, but got {type(space)}"
            )
