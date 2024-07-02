import torch

import pytest

from dlblas.autotune.space import next_power_of_2


def test_power_of_two():
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(1.5) == 1  #
    assert next_power_of_2(1) == 1
    assert next_power_of_2(1023) == 1024
