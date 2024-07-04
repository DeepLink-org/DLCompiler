import os
from dataclasses import dataclass, field

from dlblas.autotune.space import ChoiceSpace, DictSpace
from dlblas.autotune.configs import AutotuneConfig


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


class EnumerationPolicy(Policy):

    def generate(self, space):
        if isinstance(space, ChoiceSpace):
            # TODO
            return
        elif isinstance(space, DictSpace):
            # TODO
            return
        else:
            raise TypeError(
                f"space must be ChoiceSpace or DictSpace, but got {type(space)}"
            )


def get_policy(configs: AutotuneConfig):
    if configs.tunner == 'random':
        return RandomPolicy('random')
    else:
        raise NameError(f"tunner {configs.tunner} not found")
