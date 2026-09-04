"""P-25.5 handling for prohibited three-component fusion geometry.

Ordinary fusion nomenclature only relates pairs of components. When the
component-overlap graph closes a cycle, emitting another pairwise descriptor
would claim a nomenclature construction that P-25.5 explicitly prohibits.
This module proves that condition and applies the first permitted alternative:
name the corresponding carbon skeleton and restore heteroatoms by skeletal
replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..molecule import Molecule
from ..retained_fused_templates import match_retained_fused_templates, retained_parent_metadata
from ..ring_parent import RingParent
from .cover import audit_component_cover, component_scope
from .descriptor import FusionDescriptorError, build_fusion_name_ast
from .faces import FaceSearchBudgetExceeded, cached_bounded_face_model
from .model import FusionConfirmed, FusionMode, FusionNameAst, PinStatus
from .registry import fusion_component_registry
from .rules import pin_ring_size_gate

if TYPE_CHECKING:
    from collections.abc import Iterable

THIRD_COMPONENT_TIER = "p25.5-skeletal-replacement-v1"


@dataclass(frozen=True, slots=True)
class ThirdComponentFusionPlan:
    """Audited skeletal-replacement resolution of a cyclic component cover."""

    parent: RingParent
    prohibited_citation: FusionNameAst
    replacement_atom_ids: tuple[int, ...]
    cover_topology: str
    ring_sizes: tuple[int, ...]
    audit_checks: tuple[str, ...]


def plan_third_component_fusion_parent(
    mol: Molecule,
    parent_atom_ids: Iterable[int],
    *,
    mode: FusionMode | str,
) -> ThirdComponentFusionPlan | None:
    """Return a P-25.5.1 parent when ordinary pairwise fusion is impossible.

    The cyclic citation is constructed only as applicability evidence. It is
    never rendered. The emitted parent is independently planned on the
    corresponding all-carbon graph, then the original noncarbon atoms are
    restored by the established replacement-prefix operation.
    """

    atoms = frozenset(parent_atom_ids)
    policy = FusionMode(mode)
    cache_key = (
        "third_component_fusion",
        THIRD_COMPONENT_TIER,
        policy.value,
        tuple(sorted(atoms)),
    )
    if cache_key in mol._fusion_plan_cache:
        return mol._fusion_plan_cache[cache_key]
    result = _plan_uncached(mol, atoms, policy)
    mol._fusion_plan_cache[cache_key] = result
    return result


def _plan_uncached(
    mol: Molecule,
    atoms: frozenset[int],
    mode: FusionMode,
) -> ThirdComponentFusionPlan | None:
    replacement_atoms = tuple(sorted(atom for atom in atoms if mol.atoms[atom].symbol != "C"))
    if not replacement_atoms or any(mol.atoms[atom].charge for atom in atoms):
        return None
    if any(not mol.atoms[atom].element.hw_stem for atom in replacement_atoms):
        return None

    prohibited = _prohibited_pairwise_citation(mol, atoms)
    if prohibited is None:
        return None
    prohibited_ast, topology, ring_sizes = prohibited
    if mode is FusionMode.AUDITED_PIN and not pin_ring_size_gate(ring_sizes):
        return None

    carbon = _carbon_skeleton(mol, atoms)
    if not _corresponding_carbon_graph_is_exact(mol, carbon, atoms, replacement_atoms):
        return None
    parent = _carbon_parent(carbon, atoms, mode)
    if parent is None or not parent.audit_ok:
        return None
    maps = parent.proof_locant_maps
    if not maps or any(set(locant_map) != set(atoms) for locant_map in maps):
        return None
    if any(any(atom not in locant_map for atom in replacement_atoms) for locant_map in maps):
        return None

    checks = (
        "cyclic_component_cover_proved",
        "ordinary_pairwise_citation_not_emitted",
        "corresponding_carbon_graph_identity",
        "carbon_parent_audit",
        "complete_replacement_locants",
    )
    replacement_parent = parent.as_skeletal_replacement_fusion(
        replacement_atom_ids=replacement_atoms,
        audit_checks=checks,
    )
    return ThirdComponentFusionPlan(
        parent=replacement_parent,
        prohibited_citation=prohibited_ast,
        replacement_atom_ids=replacement_atoms,
        cover_topology=topology,
        ring_sizes=ring_sizes,
        audit_checks=checks,
    )


def _corresponding_carbon_graph_is_exact(
    original: Molecule,
    carbon: Molecule,
    parent_atoms: frozenset[int],
    replacement_atoms: tuple[int, ...],
) -> bool:
    """Audit carbonization and inverse skeletal replacement by graph identity."""

    replacement = frozenset(replacement_atoms)
    if replacement != {atom_id for atom_id in parent_atoms if original.atoms[atom_id].symbol != "C"}:
        return False
    if set(carbon.atoms) != set(original.atoms) or set(carbon.bonds) != set(original.bonds):
        return False
    for atom_id in parent_atoms:
        carbon_atom = carbon.atoms[atom_id]
        original_atom = original.atoms[atom_id]
        if carbon_atom.symbol != "C" or carbon_atom.charge != 0:
            return False
        restored_symbol = original_atom.symbol if atom_id in replacement else carbon_atom.symbol
        if restored_symbol != original_atom.symbol:
            return False
    for bond_id, original_bond in original.bonds.items():
        carbon_bond = carbon.bonds[bond_id]
        if (
            carbon_bond.u,
            carbon_bond.v,
            carbon_bond.order,
        ) != (
            original_bond.u,
            original_bond.v,
            original_bond.order,
        ):
            return False
    return True


def _prohibited_pairwise_citation(
    mol: Molecule,
    atoms: frozenset[int],
) -> tuple[FusionNameAst, str, tuple[int, ...]] | None:
    try:
        bounded = cached_bounded_face_model(mol, atoms)
    except FaceSearchBudgetExceeded:
        return None
    if bounded is None:
        return None
    registry = fusion_component_registry()
    matches = registry.match_faces(mol, bounded)
    try:
        ast = build_fusion_name_ast(
            mol,
            matches,
            registry,
            cover_kinds=("multiparent",),
            join_kinds=("ortho", "ortho_peri", "higher_order"),
        )
    except FusionDescriptorError:
        return None
    if ast.plan_kind != "cyclic_component_cover" or ast.citation_plan is None:
        return None
    if not ast.citation_plan.cycle_closing_join_indices:
        return None

    specs = {match.occurrence_id: registry.spec_for_match(match) for match in ast.component_occurrences}
    scopes = []
    for match in ast.component_occurrences:
        atom_by_locant = match.input_atom_by_locant
        spec = specs[match.occurrence_id]
        scopes.append(
            component_scope(
                match.occurrence_id,
                atom_by_locant.values(),
                (
                    (atom_by_locant[left], atom_by_locant[right])
                    for left, right in (bond.locants for bond in spec.bonds)
                ),
            )
        )
    target_edges = {
        tuple(sorted((bond.u, bond.v))) for bond in mol.bonds.values() if bond.u in atoms and bond.v in atoms
    }
    cover = audit_component_cover(scopes, target_atom_ids=atoms, target_edges=target_edges)
    if not cover.ok or cover.proof.topology not in {"unicyclic", "cactus"}:
        return None
    return ast, cover.proof.topology, tuple(sorted(len(face.atoms) for face in bounded.faces))


def _carbon_skeleton(mol: Molecule, parent_atoms: frozenset[int]) -> Molecule:
    carbon = Molecule()
    for atom in mol.atoms.values():
        replace = atom.idx in parent_atoms
        carbon.add_atom(
            "C" if replace else atom.symbol,
            idx=atom.idx,
            charge=0 if replace else atom.charge,
            isotope=None if replace else atom.isotope,
            stereo=atom.stereo,
            raw_stereo=atom.raw_stereo,
            cip=atom.cip,
            is_aromatic=atom.is_aromatic,
            explicit_h_count=0 if replace else atom.explicit_h_count,
            total_h_count=0 if replace else atom.total_h_count,
        )
    for bond in mol.bonds.values():
        carbon.add_bond(
            bond.u,
            bond.v,
            order=bond.order,
            idx=bond.idx,
            stereo=bond.stereo,
            in_small_ring=bond.in_small_ring,
            cip=bond.cip,
        )
    return carbon


def _carbon_parent(
    carbon: Molecule,
    atoms: frozenset[int],
    mode: FusionMode,
) -> RingParent | None:
    matches = match_retained_fused_templates(carbon, atoms)
    if not matches:
        matches = match_retained_fused_templates(
            carbon,
            atoms,
            allow_nonaromatic=True,
            pre_descriptor_only=True,
        )
    if matches:
        template = matches[0].template
        maps = [match.atom_to_locant for match in matches if match.template.name == template.name]
        base = RingParent(
            kind="retained_polycycle",
            atoms=atoms,
            retained_locant_maps=tuple(maps),
            pin_status=str(PinStatus.CONFIRMED if mode is FusionMode.AUDITED_PIN else PinStatus.VALID_GENERAL_NAME),
        )
        return base.with_retained_identity(
            name=template.name,
            locant_maps=maps,
            metadata=retained_parent_metadata(template.name),
            proof_source="p25_5_corresponding_hydrocarbon",
        )

    from .planner import plan_fusion_parent

    planned = plan_fusion_parent(carbon, atoms, mode=mode)
    if not isinstance(planned, FusionConfirmed):
        return None
    return RingParent.from_fusion_plan(planned.plan)


__all__ = [
    "THIRD_COMPONENT_TIER",
    "ThirdComponentFusionPlan",
    "plan_third_component_fusion_parent",
]
