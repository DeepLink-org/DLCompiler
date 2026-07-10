from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from triton.runtime.autotuner import Config

from .tile_shape_space import DiscreteShapeSpace


DEFAULT_UB_LIMIT_BYTES = 192 * 1024
ACQUISITION_FAILURE_WEIGHT = 0.25
ACQUISITION_EXPLORATION_WEIGHT = 0.03
ACQUISITION_UB_WEIGHT = 0.12
ACQUISITION_UB_TIE_WEIGHT = 0.01
ACQUISITION_TRUST_WEIGHT = 0.05
ACQUISITION_BOUNDARY_WEIGHT = 0.25
ACQUISITION_BATCH_DIVERSITY_WEIGHT = 0.03
UB_BARRIER_SAFE_RATIO = 0.85
_RESOURCE_OVERFLOW_RE = re.compile(
    r"requires\s+(\d+)\s+bits\s+while\s+(\d+)\s+bits\s+available",
    re.IGNORECASE,
)


ShapePoint = Tuple[int, ...]


def resource_penalty(rho: Optional[float]) -> float:
    """Smooth UB overflow penalty.

    Legal UB usage is not penalized. Once the predicted or observed usage
    exceeds the hardware limit, the first 30% overflow is soft and larger
    overflow grows quadratically.
    """
    if rho is None or rho <= 1.0:
        return 0.0
    soft_margin = 0.3
    soft_cost = 0.25
    hard_cost = 2.0
    if rho <= 1.0 + soft_margin:
        t = (rho - 1.0) / soft_margin
        return soft_cost * (t * t * (3.0 - 2.0 * t))
    t = (rho - 1.0 - soft_margin) / soft_margin
    return soft_cost + hard_cost * t * t


def resource_overflow_ratio(error: Any) -> Optional[float]:
    """Parse compiler overflow severity as required / available UB ratio."""
    text = _error_text(error)
    if not text:
        return None
    match = _RESOURCE_OVERFLOW_RE.search(text)
    if not match:
        return None
    required = int(match.group(1))
    available = int(match.group(2))
    if available <= 0:
        return None
    return required / available


def timing_value(cost: Any) -> float:
    if isinstance(cost, (list, tuple)) and cost:
        return float(cost[0])
    return float(cost)


@dataclass
class ShapeAcquisition:
    success_points: Sequence[Tuple[ShapePoint, float]]
    failure_records: Sequence[Tuple[ShapePoint, Optional[float]]]
    observed_points: Sequence[ShapePoint]
    parent_points: Sequence[ShapePoint]
    resource_model: Optional[Mapping[str, Any]]
    latency_model: Optional[Mapping[str, Any]]
    spec: object

    @classmethod
    def from_history(
        cls,
        *,
        space: DiscreteShapeSpace,
        parents: Sequence[Mapping[str, Any]],
        successes: Sequence[Mapping[str, Any]],
        failures: Sequence[Config],
        failure_observations: Sequence[Mapping[str, Any]],
        observed: Sequence[Config],
        spec: object,
    ) -> "ShapeAcquisition":
        success_points = [
            (
                space.point_from_shape(item["shape"]),
                timing_value(item["time"]),
            )
            for item in successes
        ]
        failure_records = [(space.point(config), None) for config in failures]
        failure_records.extend(
            (point, _failure_observation_resource_ratio(item))
            for point, item in (
                (_failure_observation_point(item, space), item)
                for item in failure_observations
            )
            if point is not None
        )
        observed_points = [space.point(config) for config in observed]
        parent_points = [
            space.point_from_shape(parent["shape"])
            for parent in parents
            if isinstance(parent, Mapping) and "shape" in parent
        ]
        resource_model = _fit_resource_log_model(
            _resource_observation_points(successes, failure_observations, space)
        )
        latency_model = _fit_latency_quadratic_model(success_points, spec)
        return cls(
            success_points=success_points,
            failure_records=failure_records,
            observed_points=observed_points,
            parent_points=parent_points,
            resource_model=resource_model,
            latency_model=latency_model,
            spec=spec,
        )

    def cost(self, point: ShapePoint) -> float:
        latency = _predict_latency_quadratic(point, self.latency_model, self.spec)
        novelty = _min_distance(point, self.observed_points, self.spec)
        predicted_rho = _predict_resource_rho(point, self.resource_model)
        failure = _failure_potential(point, self.failure_records, self.spec)
        ub = _resource_barrier(predicted_rho)
        ub_tie = _resource_tie_breaker(predicted_rho)
        trust = _trust_region_penalty(
            point,
            success_points=self.success_points,
            observed_points=self.observed_points,
            parent_points=self.parent_points,
            spec=self.spec,
        )
        boundary = _boundary_barrier(point, self.spec)
        return (
            latency
            + ACQUISITION_FAILURE_WEIGHT * failure
            + ACQUISITION_UB_WEIGHT * ub
            + ACQUISITION_UB_TIE_WEIGHT * ub_tie
            + ACQUISITION_TRUST_WEIGHT * trust
            + ACQUISITION_BOUNDARY_WEIGHT * boundary
            - ACQUISITION_EXPLORATION_WEIGHT * novelty
        )

    def batch_adjusted_cost(
        self,
        point: ShapePoint,
        selected_points: Sequence[ShapePoint],
    ) -> float:
        return self.cost(point) - ACQUISITION_BATCH_DIVERSITY_WEIGHT * _min_distance(
            point, selected_points, self.spec
        )


def _error_text(error: Any) -> str:
    if error is None:
        return ""
    if isinstance(error, (list, tuple)):
        return " ".join(_error_text(item) for item in error)
    if isinstance(error, BaseException):
        return f"{type(error).__name__} {error}"
    return str(error)


def _failure_observation_point(
    observation: Mapping[str, Any], space: DiscreteShapeSpace
) -> Optional[ShapePoint]:
    shape = observation.get("shape")
    if shape is None and observation.get("config") is not None:
        shape = space.shape(observation["config"])
    if not isinstance(shape, Mapping):
        return None
    try:
        return space.point_from_shape(dict(shape))
    except (KeyError, ValueError):
        return None


def _failure_observation_resource_ratio(
    observation: Mapping[str, Any],
) -> Optional[float]:
    ratio = observation.get("resource_ratio")
    if ratio is None:
        ratio = resource_overflow_ratio(observation.get("error"))
    if not isinstance(ratio, (int, float)) or ratio <= 0:
        return None
    return float(ratio)


def _resource_observation_points(
    successes: Sequence[Mapping[str, Any]],
    failure_observations: Sequence[Mapping[str, Any]],
    space: DiscreteShapeSpace,
) -> List[Tuple[ShapePoint, float]]:
    points: List[Tuple[ShapePoint, float]] = []
    for item in successes:
        ub = item.get("ub")
        if not isinstance(ub, (int, float)) or ub <= 0:
            continue
        try:
            point = space.point_from_shape(item["shape"])
        except (KeyError, ValueError):
            continue
        points.append((point, max(float(ub) / DEFAULT_UB_LIMIT_BYTES, 1e-9)))

    for item in failure_observations:
        ratio = _failure_observation_resource_ratio(item)
        if ratio is None:
            continue
        point = _failure_observation_point(item, space)
        if point is None:
            continue
        points.append((point, ratio))
    return points


def _fit_resource_log_model(
    observations: Sequence[Tuple[ShapePoint, float]],
) -> Optional[Dict[str, Any]]:
    if not observations:
        return None

    records = [
        (tuple(float(v) for v in point), math.log(max(ratio, 1e-9)))
        for point, ratio in observations
    ]
    dims = len(records[0][0])
    center = tuple(
        sum(point[dim] for point, _ in records) / len(records) for dim in range(dims)
    )
    value_center = sum(value for _, value in records) / len(records)

    gradient = [0.0 for _ in range(dims)]
    weight_sum = 0.0
    for i, (point_i, value_i) in enumerate(records):
        for point_j, value_j in records[i + 1 :]:
            delta = tuple(b - a for a, b in zip(point_i, point_j))
            norm2 = sum(value * value for value in delta)
            if norm2 <= 0.0:
                continue
            scale = (value_j - value_i) / norm2
            weight = math.sqrt(norm2)
            for dim, value in enumerate(delta):
                gradient[dim] += weight * scale * value
            weight_sum += weight

    if weight_sum > 0.0:
        gradient = [value / weight_sum for value in gradient]

    intercept = value_center - sum(g * x for g, x in zip(gradient, center))
    return {"intercept": intercept, "gradient": tuple(gradient)}


def _predict_resource_rho(
    point: ShapePoint, resource_model: Optional[Mapping[str, Any]]
) -> Optional[float]:
    if resource_model is None:
        return None
    log_rho = resource_model["intercept"] + sum(
        g * x for g, x in zip(resource_model["gradient"], point)
    )
    log_rho = min(20.0, max(-20.0, log_rho))
    return math.exp(log_rho)


def _fit_latency_quadratic_model(
    success_points: Sequence[Tuple[ShapePoint, float]],
    spec: object,
) -> Optional[Dict[str, Any]]:
    if not success_points:
        return None

    best_point, best_cost = min(success_points, key=lambda item: item[1])
    best_cost = max(float(best_cost), 1e-12)
    center = tuple(float(value) for value in best_point)
    param_count = _quadratic_feature_count(len(best_point))
    ridge = 0.03 * param_count / max(1, len(success_points))

    lhs = [[0.0 for _ in range(param_count)] for _ in range(param_count)]
    rhs = [0.0 for _ in range(param_count)]
    sigma = _kernel_sigma(spec)
    for success_point, cost in success_points:
        feature = _quadratic_features(success_point, center, spec)
        y = math.log(max(float(cost), 1e-12) / best_cost)
        distance = _index_distance(success_point, best_point, spec)
        weight = math.exp(-(distance * distance) / (2.0 * sigma * sigma))
        weight = max(weight, 0.05)
        for row in range(param_count):
            rhs[row] += weight * feature[row] * y
            for col in range(param_count):
                lhs[row][col] += weight * feature[row] * feature[col]

    for index in range(param_count):
        lhs[index][index] += ridge
    theta = _solve_linear_system(lhs, rhs)
    if theta is None:
        return None
    return {"theta": theta, "center": center}


def _predict_latency_quadratic(
    point: ShapePoint,
    model: Optional[Mapping[str, Any]],
    spec: object,
) -> float:
    if model is None:
        return 0.0
    feature = _quadratic_features(point, model["center"], spec)
    value = sum(coef * x for coef, x in zip(model["theta"], feature))
    return max(-1.0, min(2.0, value))


def _failure_potential(
    point: ShapePoint,
    failure_records: Sequence[Tuple[ShapePoint, Optional[float]]],
    spec: object,
) -> float:
    if not failure_records:
        return 0.0
    sigma = _kernel_sigma(spec)
    potential = 0.0
    for failure_point, ratio in failure_records:
        distance = _index_distance(point, failure_point, spec)
        severity = 1.0
        if ratio is not None and ratio > 1.0:
            severity += math.log(max(ratio, 1.0), 2.0)
        potential += severity * math.exp(-(distance * distance) / (2.0 * sigma * sigma))
    return potential


def _resource_barrier(rho: Optional[float]) -> float:
    if rho is None or rho <= UB_BARRIER_SAFE_RATIO:
        return 0.0
    if rho < 1.0:
        remaining = max((1.0 - rho) / (1.0 - UB_BARRIER_SAFE_RATIO), 1e-6)
        return -math.log(remaining)
    overflow = min(rho - 1.0, 16.0)
    return -math.log(1e-6) + overflow * overflow


def _resource_tie_breaker(rho: Optional[float]) -> float:
    if rho is None:
        return 0.0
    capped = max(0.0, min(rho, UB_BARRIER_SAFE_RATIO))
    return (capped / UB_BARRIER_SAFE_RATIO) ** 2


def _boundary_barrier(point: ShapePoint, spec: object) -> float:
    """Softly discourage multi-dimensional edge/corner proposals."""
    if len(point) <= 1:
        return 0.0
    denom = max(1, len(spec.values) - 1)
    if denom <= 1:
        return 0.0
    edge_values = []
    for index in point:
        t = float(index) / denom
        edge = max(0.0, (abs(2.0 * t - 1.0) - 0.70) / 0.30)
        edge_values.append(edge * edge)
    return sum(edge_values) / len(edge_values)


def _trust_region_penalty(
    point: ShapePoint,
    *,
    success_points: Sequence[Tuple[ShapePoint, float]],
    observed_points: Sequence[ShapePoint],
    parent_points: Sequence[ShapePoint],
    spec: object,
) -> float:
    if success_points:
        center = min(success_points, key=lambda item: item[1])[0]
        spread = max(
            _index_distance(center, success_point, spec)
            for success_point, _ in success_points
        )
        radius = max(0.25, min(1.0, 0.25 + spread))
    elif parent_points:
        center = parent_points[0]
        radius = 1.0
    elif observed_points:
        center = _mean_point(observed_points)
        radius = 1.0
    else:
        return 0.0
    distance = _index_distance_float(point, center, spec)
    return (distance / max(radius, 1e-6)) ** 4


def _min_distance(
    point: ShapePoint,
    points: Sequence[ShapePoint],
    spec: object,
) -> float:
    if not points:
        return 1.0
    return min(_index_distance(point, other, spec) for other in points)


def _index_distance(a: ShapePoint, b: ShapePoint, spec: object) -> float:
    return _index_distance_float(a, b, spec)


def _index_distance_float(
    a: Sequence[float],
    b: Sequence[float],
    spec: object,
) -> float:
    denom = max(1, len(spec.values) - 1)
    squared = 0.0
    for ai, bi in zip(a, b):
        squared += ((ai - bi) / denom) ** 2
    return math.sqrt(squared)


def _kernel_sigma(spec: object) -> float:
    return max(0.18, 1.5 / max(1, len(spec.values) - 1))


def _quadratic_feature_count(dims: int) -> int:
    return 1 + dims + dims + (dims * (dims - 1)) // 2


def _quadratic_features(
    point: Sequence[float],
    center: Sequence[float],
    spec: object,
) -> List[float]:
    denom = max(1, len(spec.values) - 1)
    z = [(float(value) - float(base)) / denom for value, base in zip(point, center)]
    features = [1.0]
    features.extend(z)
    features.extend(value * value for value in z)
    for left in range(len(z)):
        for right in range(left + 1, len(z)):
            features.append(z[left] * z[right])
    return features


def _solve_linear_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> Optional[List[float]]:
    size = len(vector)
    if size == 0:
        return []
    aug = [
        [float(matrix[row][col]) for col in range(size)] + [float(vector[row])]
        for row in range(size)
    ]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) <= 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for idx in range(col, size + 1):
            aug[col][idx] /= pivot_value
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) <= 1e-18:
                continue
            for idx in range(col, size + 1):
                aug[row][idx] -= factor * aug[col][idx]
    return [aug[row][size] for row in range(size)]


def _mean_point(points: Sequence[ShapePoint]) -> Tuple[float, ...]:
    dims = len(points[0])
    return tuple(
        sum(float(point[dim]) for point in points) / len(points) for dim in range(dims)
    )
