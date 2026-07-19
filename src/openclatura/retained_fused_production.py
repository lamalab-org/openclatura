"""Production gate for OPSIN-compatible retained fused derivatives.

The graph-template matcher can recognize many retained fused cores, but a core
match alone is not enough to safely name derivatives.  This module enables a
small neutral aromatic derivative set only after substituents and principal
groups have been collected, so the retained locant map is used only for
OPSIN-verified grammar classes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .assembly_parts import RetainedParentMetadata, SubstituentItem
from .grammar_snapshot_data import retained_fused_derivative_gate
from .molecule import Molecule
from .perception import PerceivedGroup
from .retained_fused_templates import RetainedFusedTemplateMatch, match_retained_fused_templates

_DERIVATIVE_GATE = retained_fused_derivative_gate()
PRODUCTION_RETAINED_FUSED_PARENTS = _DERIVATIVE_GATE.production_parent_names
ALLOWED_PRINCIPAL_KEYS = _DERIVATIVE_GATE.allowed_principal_keys
ALLOWED_GROUP_KEYS = _DERIVATIVE_GATE.allowed_group_keys
ALLOWED_SUBSTITUENT_NAMES = _DERIVATIVE_GATE.allowed_substituent_names


@dataclass(frozen=True)
class ProductionRetainedFusedParent:
    """A matched retained parent and the template metadata needed downstream."""

    name: str
    locant_maps: list[dict[int, str]]
    metadata: RetainedParentMetadata


def production_retained_fused_parent(
    mol: Molecule,
    parent_path: list[int],
    component_atoms: set[int],
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    substituent_mapping: dict[int, list[SubstituentItem]],
) -> ProductionRetainedFusedParent | None:
    """Return retained fused parent data only for verified derivative classes."""

    parent_atoms = set(parent_path)
    if not _neutral_component(mol, component_atoms):
        return None
    if not _neutral_retained_parent(mol, parent_atoms):
        return None
    if principal_key not in ALLOWED_PRINCIPAL_KEYS:
        return None
    if not _allowed_groups(parent_atoms, perceived_groups):
        return None
    if not _allowed_substituents(substituent_mapping):
        return None

    feature_atoms = set(substituent_mapping)
    for group in perceived_groups:
        if group.attachment_carbon in parent_atoms:
            feature_atoms.add(group.attachment_carbon)

    matches = [
        match
        for match in match_retained_fused_templates(
            mol,
            parent_atoms,
            include_disabled=True,
            allow_nonaromatic=principal_key == "ketone",
        )
        if match.template.name in PRODUCTION_RETAINED_FUSED_PARENTS
        and match.template.derivative_production_enabled
        and (principal_key != "ketone" or _has_mancude_unsaturation(mol, parent_atoms, match))
    ]
    if not matches:
        return None

    parent_name = matches[0].template.name
    maps = [
        match.atom_to_locant
        for match in matches
        if match.template.name == parent_name
        and _feature_locants_are_substitutable(match.atom_to_locant, feature_atoms)
    ]
    if not maps:
        return None
    template = matches[0].template
    return ProductionRetainedFusedParent(
        name=parent_name,
        locant_maps=maps,
        metadata=RetainedParentMetadata(
            default_indicated_h=template.default_indicated_h,
            fusion_locants=template.fusion_atoms,
            derivative_stem=template.derivative_stem,
        ),
    )


def _neutral_component(mol: Molecule, atoms: set[int]) -> bool:
    return all(mol.atoms[atom].charge == 0 for atom in atoms)


def _has_mancude_unsaturation(
    mol: Molecule,
    parent_atoms: set[int],
    match: RetainedFusedTemplateMatch,
) -> bool:
    """Reject hydro derivatives that merely share an oxo-parent topology."""

    expected_double_bonds = match.template.mancude_double_bonds
    if expected_double_bonds is None:
        return False
    nonaromatic_parent_carbonyls = sum(
        atom_template.symbol == "C"
        and not atom_template.aromatic
        and any(
            neighbor not in parent_atoms
            and mol.atoms[neighbor].symbol == "O"
            and (bond := mol.get_bond(match.locant_to_atom[atom_template.locant], neighbor)) is not None
            and bond.order == 2
            for neighbor in mol.get_neighbors(match.locant_to_atom[atom_template.locant])
        )
        for atom_template in match.template.atoms
    )
    actual_double_bonds = sum(
        bond.order == 2 and (bond.u in parent_atoms or bond.v in parent_atoms) for bond in mol.bonds.values()
    )
    return actual_double_bonds >= expected_double_bonds + nonaromatic_parent_carbonyls


def _neutral_retained_parent(mol: Molecule, atoms: set[int]) -> bool:
    if any(mol.atoms[atom].charge != 0 for atom in atoms):
        return False
    for atom in atoms:
        for neighbor in mol.get_neighbors(atom):
            if neighbor in atoms:
                bond = mol.get_bond(atom, neighbor)
                if bond is None or bond.order not in {1, 2}:
                    return False
    return True


def _allowed_groups(parent_atoms: set[int], perceived_groups: list[PerceivedGroup]) -> bool:
    for group in perceived_groups:
        if group.attachment_carbon not in parent_atoms:
            continue
        if group.key not in ALLOWED_GROUP_KEYS:
            return False
    return True


def _allowed_substituents(substituent_mapping: dict[int, list[SubstituentItem]]) -> bool:
    for items in substituent_mapping.values():
        for item in items:
            if item.spiro is not None:
                return False
            if item.name not in ALLOWED_SUBSTITUENT_NAMES:
                return False
    return True


def _feature_locants_are_substitutable(atom_to_locant: dict[int, str], feature_atoms: set[int]) -> bool:
    for atom in feature_atoms:
        locant = atom_to_locant.get(atom)
        if locant is None or any(char.isalpha() for char in locant):
            return False
    return True
