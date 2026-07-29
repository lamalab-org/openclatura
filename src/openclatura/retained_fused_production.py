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
from .namer_config import INDICATED_H_RETAINED_NAMES
from .perception import PerceivedGroup
from .retained_fused_templates import RetainedFusedTemplateMatch, match_retained_fused_templates

_DERIVATIVE_GATE = retained_fused_derivative_gate()
PRODUCTION_RETAINED_FUSED_PARENTS = _DERIVATIVE_GATE.production_parent_names
ALLOWED_PRINCIPAL_KEYS = _DERIVATIVE_GATE.allowed_principal_keys
ALLOWED_GROUP_KEYS = _DERIVATIVE_GATE.allowed_group_keys


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
    if not _added_hydrogen_is_citable(mol, parent_atoms, template):
        return None
    return ProductionRetainedFusedParent(
        name=parent_name,
        locant_maps=maps,
        metadata=RetainedParentMetadata(
            default_indicated_h=template.default_indicated_h,
            fusion_locants=template.fusion_atoms,
            derivative_stem=template.derivative_stem,
            indicated_hydrogen_count=_indicated_hydrogen_count(template),
        ),
    )


def _added_hydrogen_is_citable(mol: Molecule, parent_atoms: set[int], template) -> bool:
    """Whether every saturated position of the parent can be spelt out.

    A lactam such as 4-oxoquinoline-3-carboxamide saturates two ring positions
    beyond what quinoline supports, and which two cannot be recovered from the
    name -- OPSIN puts the hydrogen on C3 and reads a different molecule.  They
    have to be cited, ``4-oxo-1,4-dihydroquinoline-3-carboxamide``, and only
    parents wired into the indicated-hydrogen machinery can do that.  A parent
    needing an uncitable added hydrogen falls back to von Baeyer, which states
    its saturation positionally and so stays unambiguous.
    """

    saturated = 0
    for atom in parent_atoms:
        ring_bonds = [
            bond
            for neighbor in mol.get_neighbors(atom)
            if neighbor in parent_atoms and (bond := mol.get_bond(atom, neighbor)) is not None
        ]
        if ring_bonds and sum(bond.order for bond in ring_bonds) == len(ring_bonds):
            saturated += 1
    if saturated <= _indicated_hydrogen_count(template):
        return True
    return template.name in INDICATED_H_RETAINED_NAMES


def _indicated_hydrogen_count(template) -> int:
    """How many indicated hydrogens this mancude parent hydride supports.

    A parent that declares its own positions (1H-indole, 9H-xanthene) supports
    exactly that many.  Otherwise the mancude double-bond count leaves the
    skeletal atoms it cannot pair: purine's nine atoms and four double bonds
    leave one, which is why purine is cited as 7H- or 9H-.  A parent with
    neither is fully mancude and supports none.
    """

    if template.default_indicated_h:
        return len(template.default_indicated_h)
    if template.mancude_double_bonds:
        return len(template.atoms) - 2 * template.mancude_double_bonds
    return 0


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
    """Whether the substituents on a retained parent are ones we can render.

    A substituent is named by its own recursive call, so what it *is* cannot
    affect whether the parent's locants are right; only a spiro junction, which
    the retained renderer cannot compose, disqualifies the parent here.
    """

    return all(item.spiro is None for items in substituent_mapping.values() for item in items)


def _feature_locants_are_substitutable(atom_to_locant: dict[int, str], feature_atoms: set[int]) -> bool:
    for atom in feature_atoms:
        locant = atom_to_locant.get(atom)
        if locant is None or any(char.isalpha() for char in locant):
            return False
    return True
