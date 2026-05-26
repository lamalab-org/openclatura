"""Functional-group metadata and perception extension points."""

from collections.abc import Callable
from dataclasses import dataclass

from .molecule import FunctionalGroupMetadata, Molecule
from .nomenclature import RULES

PERCEPTION_DETECTORS: list[Callable[[Molecule], list]] = []


@dataclass(frozen=True)
class PerceptionDetectorSpec:
    key: str
    detector: Callable[[Molecule], list]
    priority: int = 100
    families: tuple[str, ...] = ()
    description: str = ""


PERCEPTION_SPECS: list[PerceptionDetectorSpec] = []


def register_group_detector(
    detector: Callable[[Molecule], list], *, prepend: bool = False
) -> Callable[[Molecule], list]:
    """Register a detector that returns raw ``PerceivedGroup`` objects."""

    if prepend:
        PERCEPTION_DETECTORS.insert(0, detector)
    else:
        PERCEPTION_DETECTORS.append(detector)
    return detector


def register_perception_spec(spec: PerceptionDetectorSpec, *, prepend: bool = False) -> PerceptionDetectorSpec:
    """Register an ordered detector spec for custom functional-group perception."""

    if prepend:
        PERCEPTION_SPECS.insert(0, spec)
    else:
        PERCEPTION_SPECS.append(spec)
    PERCEPTION_SPECS.sort(key=lambda item: item.priority)
    return spec


def metadata_for_group(key: str) -> FunctionalGroupMetadata:
    """Return metadata for built-in, overridden, or custom groups."""

    rule = RULES.functional_groups.by_key.get(key)
    if rule is None:
        return FunctionalGroupMetadata(source="perception")
    return FunctionalGroupMetadata(
        prefix=rule.prefix,
        suffix=rule.suffix,
        multi_suffix=rule.multi_suffix,
        suffix_multiplier_positions=rule.suffix_multiplier_positions,
        seniority=rule.seniority,
        suffix_with_locant=rule.suffix_with_locant,
        source="nomenclature.functional_groups",
    )
