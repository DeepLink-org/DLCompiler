import os

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AutotuneConfig:
    # iteration for tunning
    total_iteration: int = 5

    # tunner
    tunner: str = 'enumeration'
