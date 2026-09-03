"""Validated, lazily loaded policy for systematic fusion nomenclature.

The shared retained-template registry owns component graphs and the element
registry owns atom chemistry. This module contains only fusion-specific policy,
deterministic search limits, and intrinsic layout vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from ..naming_data import load_json_table


@dataclass(frozen=True, slots=True)
class FusionRuleConfig:
    planner_tier: str
    pin_minimum_ring_size: int
    pin_minimum_ring_count: int


@dataclass(frozen=True, slots=True)
class FusionSearchLimits:
    minimum_ring_size: int
    maximum_ring_size: int
    cycle_states: int
    face_model_states: int
    layout_states: int
    maximum_layouts: int
    maximum_component_occurrences: int
    maximum_component_selections: int
    component_selection_states: int
    locant_map_combinations: int
    mancude_states: int


@dataclass(frozen=True, slots=True)
class RingShapeSpec:
    ring_size: int
    shape_id: str
    vertices: tuple[tuple[int, int], ...]
    edge_directions: tuple[int, ...]
    horizontal_axis_class: str
    distortion_rank: int = 0

    def __post_init__(self) -> None:
        if not 3 <= self.ring_size <= 8:
            raise ValueError("intrinsic ring shapes support sizes 3 through 8")
        if len(self.vertices) != self.ring_size or len(set(self.vertices)) != self.ring_size:
            raise ValueError("ring shape vertices must uniquely cover the declared ring size")
        if len(self.edge_directions) != self.ring_size:
            raise ValueError("ring shape needs one direction class per edge")
        if self.vertices[:2] != ((0, 0), (4, 0)):
            raise ValueError("ring shapes must use the canonical four-unit entrance edge")
        if self.distortion_rank < 0:
            raise ValueError("shape distortion rank must be non-negative")


@dataclass(frozen=True, slots=True)
class FusionNomenclatureConfig:
    registry_version: str
    graph_source: str
    rules: FusionRuleConfig
    search: FusionSearchLimits
    ring_shapes: tuple[RingShapeSpec, ...]


@cache
def fusion_nomenclature_config() -> FusionNomenclatureConfig:
    return fusion_nomenclature_config_from_data(load_json_table("fusion_components.json"))


def fusion_nomenclature_config_from_data(data: dict) -> FusionNomenclatureConfig:
    """Validate one complete fusion configuration mapping."""

    if data.get("schema_version") != 1:
        raise ValueError("unsupported fusion nomenclature schema version")
    graph_source = _text(data, "graph_source")
    if graph_source != "retained_graph_templates":
        raise ValueError("fusion components must use the shared retained graph-template registry")
    rules_data = _mapping(data, "rules")
    pin_gate = _mapping(rules_data, "pin_gate")
    rules = FusionRuleConfig(
        planner_tier=_text(rules_data, "planner_tier"),
        pin_minimum_ring_size=_positive_int(pin_gate, "minimum_ring_size"),
        pin_minimum_ring_count=_positive_int(pin_gate, "minimum_ring_count"),
    )
    limits_data = _mapping(data, "search_limits")
    search = FusionSearchLimits(
        **{field: _positive_int(limits_data, field) for field in FusionSearchLimits.__dataclass_fields__}
    )
    if search.minimum_ring_size > search.maximum_ring_size:
        raise ValueError("fusion minimum ring size must not exceed the maximum")
    rows = data.get("ring_shapes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("fusion ring_shapes must be a non-empty list")
    shapes = tuple(_ring_shape(row) for row in rows)
    if len({shape.shape_id for shape in shapes}) != len(shapes):
        raise ValueError("fusion ring shape ids must be unique")
    return FusionNomenclatureConfig(
        registry_version=_text(data, "registry_version"),
        graph_source=graph_source,
        rules=rules,
        search=search,
        ring_shapes=shapes,
    )


def _ring_shape(row: object) -> RingShapeSpec:
    if not isinstance(row, dict):
        raise ValueError("every fusion ring shape must be a mapping")
    return RingShapeSpec(
        ring_size=_positive_int(row, "ring_size"),
        shape_id=_text(row, "id"),
        vertices=tuple(_point(value) for value in row.get("vertices", ())),
        edge_directions=tuple(_integer(value, "edge direction") for value in row.get("edge_directions", ())),
        horizontal_axis_class=_text(row, "horizontal_axis_class"),
        distortion_rank=_nonnegative_int(row, "distortion_rank", default=0),
    )


def _mapping(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"fusion nomenclature {key} must be a mapping")
    return value


def _text(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"fusion nomenclature {key} must be non-empty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"fusion nomenclature {label} must be an integer")
    return value


def _positive_int(data: dict, key: str) -> int:
    value = _integer(data.get(key), key)
    if value <= 0:
        raise ValueError(f"fusion nomenclature {key} must be positive")
    return value


def _nonnegative_int(data: dict, key: str, *, default: int | None = None) -> int:
    value = _integer(data.get(key, default), key)
    if value < 0:
        raise ValueError(f"fusion nomenclature {key} must be non-negative")
    return value


def _point(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("fusion ring shape vertices must be coordinate pairs")
    return (_integer(value[0], "x coordinate"), _integer(value[1], "y coordinate"))
