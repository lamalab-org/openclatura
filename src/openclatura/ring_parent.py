"""Structured ring-parent handoff objects.

`RingSystem` is the legacy discovery object.  `RingParent` is the migration
target: one object that keeps descriptor text, numbering candidates, locant
maps, and audit metadata together instead of passing a descriptor string and
plain paths independently.
"""

from dataclasses import dataclass

from .fusion.model import FusionParentPlan
from .locants import retained_locant_sort_key
from .polycycle_topology import RingNumbering
from .ring_renderer import is_von_baeyer_descriptor


@dataclass(frozen=True)
class RingParent:
    kind: str
    atoms: frozenset[int]
    descriptor: str | None = None
    descriptor_numbers: tuple[int, ...] = ()
    candidate_paths: tuple[tuple[int, ...], ...] = ()
    numbering_candidates: tuple[RingNumbering, ...] = ()
    selected_numbering: RingNumbering | None = None
    retained_locant_maps: tuple[dict[int, str], ...] = ()
    proof_source: str = "topology"
    fusion_plan: FusionParentPlan | None = None

    @property
    def paths(self) -> list[list[int]]:
        if self.numbering_candidates:
            return [list(numbering.path) for numbering in self.numbering_candidates]
        if self.retained_locant_maps:
            return [
                sorted(locant_map, key=lambda atom: retained_locant_sort_key(locant_map[atom]))
                for locant_map in self.retained_locant_maps
            ]
        return [list(path) for path in self.candidate_paths]

    @property
    def locant_map(self) -> dict[int, str] | None:
        if self.selected_numbering is None:
            return None
        return self.selected_numbering.locant_map

    @property
    def audit_ok(self) -> bool:
        if self.kind == "systematic_fusion":
            return self.fusion_plan is not None and self.fusion_plan.audit.confirmed
        if self.retained_locant_maps:
            return all(
                set(locant_map) == set(self.atoms) and len(set(locant_map.values())) == len(self.atoms)
                for locant_map in self.retained_locant_maps
            )
        if not self.numbering_candidates:
            return not is_von_baeyer_descriptor(self.descriptor)
        return all(numbering.audit_ok for numbering in self.numbering_candidates)

    @classmethod
    def from_fusion_plan(cls, plan: FusionParentPlan) -> "RingParent":
        """Build a ring-parent handoff only from an independently audited plan."""

        if not plan.audit.confirmed:
            raise ValueError("Systematic fusion RingParent requires a confirmed audit.")
        maps = plan.numbering.string_input_locant_maps()
        return cls(
            kind="systematic_fusion",
            atoms=frozenset(dict(maps[0])) if maps else frozenset(),
            descriptor=plan.rendered_base_name,
            retained_locant_maps=maps,
            proof_source="fusion_reconstruction",
            fusion_plan=plan,
        )

    @classmethod
    def from_numberings(
        cls,
        *,
        kind: str,
        atoms: set[int] | frozenset[int],
        descriptor: str | None,
        descriptor_numbers: tuple[int, ...],
        numberings: list[RingNumbering],
        selected_path: list[int] | tuple[int, ...] | None = None,
    ) -> "RingParent":
        selected = None
        if selected_path is not None:
            selected_tuple = tuple(selected_path)
            selected = next((numbering for numbering in numberings if numbering.path == selected_tuple), None)
        if selected is None and numberings:
            selected = numberings[0]
        return cls(
            kind=kind,
            atoms=frozenset(atoms),
            descriptor=descriptor,
            descriptor_numbers=descriptor_numbers,
            candidate_paths=tuple(numbering.path for numbering in numberings),
            numbering_candidates=tuple(numberings),
            selected_numbering=selected,
            proof_source="descriptor",
        )

    @classmethod
    def from_paths(
        cls,
        *,
        kind: str,
        atoms: set[int] | frozenset[int],
        descriptor: str | None,
        paths: list[list[int]] | tuple[tuple[int, ...], ...],
        descriptor_numbers: tuple[int, ...] = (),
    ) -> "RingParent":
        if is_von_baeyer_descriptor(descriptor):
            raise ValueError("von Baeyer RingParent requires audited numbering candidates")
        return cls(
            kind=kind,
            atoms=frozenset(atoms),
            descriptor=descriptor,
            descriptor_numbers=descriptor_numbers,
            candidate_paths=tuple(tuple(path) for path in paths),
        )

    @classmethod
    def from_retained_locant_maps(
        cls,
        *,
        atoms: set[int] | frozenset[int],
        locant_maps: list[dict[int, str]],
    ) -> "RingParent":
        """Build an audited parent proof from exact template isomorphisms."""

        parent = cls(
            kind="retained_polycycle",
            atoms=frozenset(atoms),
            retained_locant_maps=tuple(dict(locant_map) for locant_map in locant_maps),
            proof_source="retained_template",
        )
        if not parent.audit_ok:
            raise ValueError("Retained parent locant maps must be complete bijections.")
        return parent
