"""Structured ring-parent handoff objects.

`RingSystem` is the legacy discovery object.  `RingParent` is the migration
target: one object that keeps descriptor text, numbering candidates, locant
maps, and audit metadata together instead of passing a descriptor string and
plain paths independently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from .locants import retained_locant_sort_key
from .polycycle_topology import RingNumbering
from .ring_renderer import is_von_baeyer_descriptor

if TYPE_CHECKING:
    from .fusion.model import FusionParentPlan, ParentBondModel
    from .fusion.wrappers import BridgedFusionWrapperPlan


class ParentHydrideKind(str, Enum):
    """Naming system that supplies a selected ring parent hydride."""

    GENERATED_MONOCYCLE = "generated_monocycle"
    RETAINED = "retained"
    SYSTEMATIC_FUSION = "systematic_fusion"
    BRIDGED_FUSION = "bridged_fusion"
    VON_BAEYER = "von_baeyer"
    SPIRO = "spiro"
    LEGACY_RING = "legacy_ring"


@dataclass(frozen=True)
class ParentHydrideMetadata:
    """Hydrogen and bond-capacity facts implied by a named parent hydride."""

    default_indicated_h: tuple[str, ...] = ()
    fusion_locants: tuple[str, ...] = ()
    derivative_stem: str | None = None
    indicated_hydrogen_count: int = 0
    mancude_double_bonds: int = 0
    relocated_indicated_h: bool = False
    inherent_saturated_locants: tuple[str, ...] = ()


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
    fusion_wrapper_plan: BridgedFusionWrapperPlan | None = None
    parent_hydride_kind: ParentHydrideKind | None = None
    parent_name: str | None = None
    substituent_stem: str | None = None
    hydride_metadata: ParentHydrideMetadata | None = None
    parent_bond_model: ParentBondModel | None = None
    pin_status: str = "unknown"

    @property
    def hydride_kind(self) -> ParentHydrideKind:
        """Return the naming system without changing the legacy topology kind."""

        if self.parent_hydride_kind is not None:
            return self.parent_hydride_kind
        if self.is_systematic_fusion:
            return ParentHydrideKind.SYSTEMATIC_FUSION
        if self.is_bridged_fusion:
            return ParentHydrideKind.BRIDGED_FUSION
        if self.retained_locant_maps and self.kind == "retained_polycycle":
            return ParentHydrideKind.RETAINED
        if self.kind == "bicyclo" or is_von_baeyer_descriptor(self.descriptor):
            return ParentHydrideKind.VON_BAEYER
        if self.kind in {"spiro", "dispiro"}:
            return ParentHydrideKind.SPIRO
        if self.kind == "monocycle":
            return ParentHydrideKind.GENERATED_MONOCYCLE
        return ParentHydrideKind.LEGACY_RING

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
        if self.is_bridged_fusion:
            return self.fusion_wrapper_plan is not None and self.fusion_wrapper_plan.audit_ok
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

    @property
    def is_systematic_fusion(self) -> bool:
        return self.kind == "systematic_fusion" and self.fusion_plan is not None

    @property
    def is_bridged_fusion(self) -> bool:
        return self.kind == "bridged_fusion" and self.fusion_wrapper_plan is not None

    @property
    def is_fusion_parent(self) -> bool:
        """Whether fusion nomenclature supplied the complete parent hydride."""

        return self.is_systematic_fusion or self.is_bridged_fusion

    @property
    def absorbs_skeletal_replacement(self) -> bool:
        """Whether heteroatom replacement is already present in ``base_name``."""

        return self.hydride_kind in {
            ParentHydrideKind.RETAINED,
            ParentHydrideKind.SYSTEMATIC_FUSION,
            ParentHydrideKind.BRIDGED_FUSION,
        }

    @property
    def base_name(self) -> str | None:
        if self.parent_name is not None:
            return self.parent_name
        if self.is_systematic_fusion:
            return self.fusion_plan.rendered_base_name
        if self.is_bridged_fusion:
            return self.fusion_wrapper_plan.rendered_name
        from .rules import stems

        stem = stems.stem_for(len(self.atoms))
        if self.descriptor is not None:
            return f"{self.descriptor}{stem}ane"
        if self.hydride_kind is ParentHydrideKind.GENERATED_MONOCYCLE:
            return f"cyclo{stem}ane"
        return None

    @property
    def binding_term(self) -> str | None:
        """Return parent text that survives suffix and unsaturation morphology."""

        if self.hydride_kind in {
            ParentHydrideKind.RETAINED,
            ParentHydrideKind.SYSTEMATIC_FUSION,
            ParentHydrideKind.BRIDGED_FUSION,
        }:
            return self.base_name
        return self.descriptor

    @property
    def derivative_stem(self) -> str | None:
        return self.substituent_stem

    def base_hydride_name(self, parent_length: int) -> str | None:
        """Return the saturated parent-hydride spelling for diagnostics."""

        if parent_length != len(self.atoms):
            return None
        return self.base_name

    def assembly_stem_and_terminal(self, parent_length: int) -> tuple[str, str] | None:
        """Project the hydride onto the existing suffix assembler."""

        if self.hydride_kind in {
            ParentHydrideKind.SYSTEMATIC_FUSION,
            ParentHydrideKind.BRIDGED_FUSION,
            ParentHydrideKind.RETAINED,
        }:
            name = self.base_name
            if name is None:
                return None
            return (name[:-1], "e") if name.endswith("e") else (name, "")
        from .rules import stems

        stem = stems.stem_for(parent_length)
        if self.descriptor is not None:
            return f"{self.descriptor}{stem}", "e"
        if self.hydride_kind is ParentHydrideKind.GENERATED_MONOCYCLE:
            return f"cyclo{stem}", "e"
        return None

    @property
    def bond_model(self) -> ParentBondModel | None:
        if self.parent_bond_model is not None:
            return self.parent_bond_model
        return self.fusion_plan.bond_model if self.is_systematic_fusion else None

    @property
    def proof_locant_maps(self) -> tuple[dict[int, str], ...]:
        if self.is_systematic_fusion:
            return self.fusion_plan.numbering.string_input_locant_maps()
        if self.is_bridged_fusion:
            return tuple(dict(entries) for entries in self.fusion_wrapper_plan.parent.locant_maps)
        if self.retained_locant_maps:
            return self.retained_locant_maps
        return tuple(
            numbering.locant_map for numbering in self.numbering_candidates if numbering.audit_ok
        )

    @property
    def metadata(self) -> ParentHydrideMetadata | None:
        """Hydride metadata used by additive and subtractive operations."""

        return self.hydride_metadata

    @property
    def retained_name(self) -> str | None:
        """Compatibility view for code that specifically needs retained rules."""

        return self.base_name if self.hydride_kind is ParentHydrideKind.RETAINED else None

    @property
    def implies_parent_unsaturation(self) -> bool:
        """Whether unsaturation is supplied by the named parent hydride."""

        return self.hydride_kind in {
            ParentHydrideKind.RETAINED,
            ParentHydrideKind.SYSTEMATIC_FUSION,
            ParentHydrideKind.BRIDGED_FUSION,
        }

    @property
    def numbering_uses_proof_locant_maps(self) -> bool:
        """Whether the existing numbering scorer must consume proof maps."""

        return self.hydride_kind in {
            ParentHydrideKind.RETAINED,
            ParentHydrideKind.SYSTEMATIC_FUSION,
            ParentHydrideKind.BRIDGED_FUSION,
        } or is_von_baeyer_descriptor(self.descriptor)

    def with_retained_identity(
        self,
        *,
        name: str,
        locant_maps: list[dict[int, str]] | tuple[dict[int, str], ...] | None,
        metadata: ParentHydrideMetadata | None,
        proof_source: str = "retained_template",
    ) -> RingParent:
        """Return this topology enriched by a retained-parent proof."""

        maps = tuple(dict(locant_map) for locant_map in (locant_maps or ()))
        enriched = replace(
            self,
            parent_hydride_kind=ParentHydrideKind.RETAINED,
            parent_name=name,
            substituent_stem=metadata.derivative_stem if metadata is not None else None,
            retained_locant_maps=maps,
            hydride_metadata=metadata,
            proof_source=proof_source,
        )
        if maps and not enriched.audit_ok:
            raise ValueError("Retained parent locant maps must be complete bijections.")
        return enriched

    @classmethod
    def from_fusion_plan(cls, plan: FusionParentPlan) -> RingParent:
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
            parent_hydride_kind=ParentHydrideKind.SYSTEMATIC_FUSION,
            parent_name=plan.rendered_base_name,
            hydride_metadata=ParentHydrideMetadata(
                default_indicated_h=tuple(str(locant) for locant in plan.indicated_hydrogens),
                fusion_locants=tuple(
                    str(locant)
                    for _, locant in plan.numbering.abstract_atom_to_locant
                    if locant.fusion_suffix or locant.interior_distance is not None
                ),
                indicated_hydrogen_count=len(plan.indicated_hydrogens),
                mancude_double_bonds=plan.bond_model.maximum_non_cumulative_double_bonds,
            ),
            parent_bond_model=plan.bond_model,
            pin_status=str(plan.pin_status),
        )

    @classmethod
    def from_fusion_wrapper(cls, plan: BridgedFusionWrapperPlan) -> RingParent:
        """Build a parent handoff from an audited nondetachable bridge plan."""

        if not plan.audit_ok:
            raise ValueError("Bridged fusion RingParent requires a confirmed wrapper audit.")
        underlying_fusion = plan.parent.fusion_plan
        all_atoms = set(plan.parent.atom_ids)
        all_atoms.update(atom for bridge in plan.bridges for atom in bridge.atom_ids)
        return cls(
            kind="bridged_fusion",
            atoms=frozenset(all_atoms),
            descriptor=plan.rendered_name,
            retained_locant_maps=tuple(dict(entries) for entries in plan.parent.locant_maps),
            proof_source="fusion_wrapper_reconstruction",
            fusion_plan=underlying_fusion,
            fusion_wrapper_plan=plan,
            parent_hydride_kind=ParentHydrideKind.BRIDGED_FUSION,
            parent_name=plan.rendered_name,
            parent_bond_model=(underlying_fusion.bond_model if underlying_fusion is not None else None),
            pin_status=(
                str(underlying_fusion.pin_status)
                if underlying_fusion is not None
                else "valid_general_name"
            ),
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
    ) -> RingParent:
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
    ) -> RingParent:
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
        name: str | None = None,
        metadata: ParentHydrideMetadata | None = None,
    ) -> RingParent:
        """Build an audited parent proof from exact template isomorphisms."""

        parent = cls(
            kind="retained_polycycle",
            atoms=frozenset(atoms),
            retained_locant_maps=tuple(dict(locant_map) for locant_map in locant_maps),
            proof_source="retained_template",
            parent_hydride_kind=ParentHydrideKind.RETAINED,
            parent_name=name,
            substituent_stem=metadata.derivative_stem if metadata is not None else None,
            hydride_metadata=metadata,
        )
        if not parent.audit_ok:
            raise ValueError("Retained parent locant maps must be complete bijections.")
        return parent


# ``RingParent`` remains the public compatibility name.  New pipeline code uses
# the semantic alias to make clear that the object is the complete handoff to
# numbering and assembly, not merely a discovered ring topology.
ParentHydridePlan = RingParent
