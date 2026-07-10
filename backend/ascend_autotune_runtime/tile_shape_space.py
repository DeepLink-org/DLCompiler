from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from triton.runtime.autotuner import Config


Shape = Dict[str, int]
ShapeKey = Tuple[int, ...]
ShapePoint = Tuple[int, ...]


def shape_key(shape: Shape, params: Sequence[str]) -> ShapeKey:
    return tuple(shape[name] for name in params)


def extract_shape(config: Config, params: Sequence[str]) -> Shape:
    return {name: config.kwargs[name] for name in params}


def make_shape_config(
    base_config: Config, shape: Shape, params: Sequence[str]
) -> Config:
    kwargs = dict(base_config.kwargs)
    for name in params:
        kwargs.pop(name, None)
    kwargs.update(shape)
    return Config(
        kwargs=kwargs,
        num_warps=getattr(base_config, "num_warps", 4),
        num_stages=getattr(base_config, "num_stages", 2),
        num_ctas=getattr(base_config, "num_ctas", 1),
        maxnreg=getattr(base_config, "maxnreg", None),
        pre_hook=getattr(base_config, "pre_hook", None),
        ir_override=getattr(base_config, "ir_override", None),
    )


def expand_search_param_shapes(base_config: Config, spec) -> List[Config]:
    import itertools

    return [
        make_shape_config(base_config, dict(zip(spec.params, combo)), spec.params)
        for combo in itertools.product(spec.values, repeat=len(spec.params))
    ]


def effective_search_value_map(configs: Sequence[Config], spec) -> Dict[str, List[int]]:
    return {
        name: sorted(
            {config.kwargs[name] for config in configs if name in config.kwargs}
        )
        for name in spec.params
    }


@dataclass(frozen=True)
class DiscreteShapeSpace:
    """Index-space view of search-param configs.

    Search algorithms operate on small integer index vectors, while Triton
    kernels need concrete ``Config`` objects. This class keeps that translation
    in one place.
    """

    configs: Sequence[Config]
    spec: object

    def __post_init__(self):
        by_key = {}
        by_point = {}
        unique = []
        for config in self.configs:
            shape = self.shape(config)
            key = shape_key(shape, self.spec.params)
            if key not in by_key:
                by_key[key] = config
                unique.append(config)
            point = self.point_from_shape(shape)
            by_point.setdefault(point, config)
        object.__setattr__(self, "by_key", by_key)
        object.__setattr__(self, "by_point", by_point)
        object.__setattr__(self, "unique_configs", tuple(unique))
        object.__setattr__(self, "points", tuple(by_point))
        object.__setattr__(
            self, "value_map", effective_search_value_map(self.configs, self.spec)
        )
        object.__setattr__(
            self,
            "values_by_dim",
            tuple(
                sorted({point[dim] for point in by_point})
                for dim in range(len(self.spec.params))
            ),
        )

    def shape(self, config: Config) -> Shape:
        return extract_shape(config, self.spec.params)

    def key(self, config: Config) -> ShapeKey:
        return shape_key(self.shape(config), self.spec.params)

    def point(self, config: Config) -> ShapePoint:
        return self.point_from_shape(self.shape(config))

    def point_from_shape(self, shape: Shape) -> ShapePoint:
        return tuple(self.spec.values.index(shape[name]) for name in self.spec.params)

    def config_for_point(self, point: ShapePoint) -> Config | None:
        return self.by_point.get(point)

    def contains_key(self, key: ShapeKey) -> bool:
        return key in self.by_key
