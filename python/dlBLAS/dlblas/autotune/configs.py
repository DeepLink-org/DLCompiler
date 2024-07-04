import os

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AutotuneConfig:
    # iteration for tunning
    iteration: int = 10

    # tunner
    tunner: str = 'random'
