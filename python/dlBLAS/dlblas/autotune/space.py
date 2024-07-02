import math
import random
from dataclasses import dataclass, field, astuple
from typing import Any


def next_power_of_2(n):
    """
    Given a positive integer n, this function returns the smallest power of 2 
    (2^x) that is greater than or equal to n.
    
    :param n: A positive integer.
    :return: The next power of 2 greater than or equal to n.
    """
    if isinstance(n, float):
        n = int(n)
    assert isinstance(n,
                      int) and n > 0, f'expect positive integer, but got {n}'

    # If the number is already a power of 2, return it directly
    if n & (n - 1) == 0:
        return n

    # Otherwise, calculate the next power of 2
    power = 1
    while power < n:
        power <<= 1

    return power


@dataclass(frozen=True)
class Space:

    def sample(self):
        raise NotImplementedError()


@dataclass(frozen=True)
class FixedSpace(Space):
    value: Any

    def sample(self):
        return self.value


@dataclass(frozen=True)
class RangeSapce(Space):
    start: float
    end: float

    def __post_init__(self):
        assert self.end > self.start
        self.scale = self.end - self.start

    def sample(self):
        rand = random.uniform(0, 1)
        return self.start + rand * self.scale


@dataclass(frozen=True)
class DiscreteSpace(Space):
    start: int
    end: int

    def __post_init__(self):
        assert self.end > self.start

    def sample(self):
        return random.randint(self.start, self.end)


@dataclass(frozen=True)
class PowerOfTwoSpace(DiscreteSpace):

    def __post_init__(self):
        super().__post_init__()
        self.start = next_power_of_2(self.start)
        self.end = next_power_of_2(self.end)
        self.start_base = int(math.log2(self.start))
        self.end_base = int(math.log2(self.end))

    def sample(self):
        n = random.randint(self.start_base, self.end_base)
        return 2**n


@dataclass(frozen=True)
class DictSpace:
    params: dict

    def __post_init__(self):
        assert len(self.params) > 0, f'empty params: {self.params}'
        for k, v in self.params.items():
            assert isinstance(v, Space)
            assert not isinstance(v, ChoiceSpace)
            assert not isinstance(v, DictSpace)

    def sample(self):
        ans = {}
        for k, v in self.params.items():
            ans[k] = v.sample()
        return ans


@dataclass(frozen=True)
class ChoiceSpace:
    choices: list

    def __post_init__(self):
        assert len(self.choices) > 0, f'empty choices: {self.choices}'

    def sample(self):
        return random.choice(self.choices)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.choices[index]
        else:
            raise TypeError("Index must be an integer")
