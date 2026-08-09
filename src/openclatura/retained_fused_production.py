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
    attachment_atom: int | None = None,
) -> ProductionRetainedFusedParent | None:
    """Return retained fused parent data only for verified derivative classes.

    ``attachment_atom`` is the ring atom a substituent prefix hangs from; it is
    a feature locant just like a substituted position, so the retained map has
    to be able to cite it (``quinolin-7-yl``, never a fusion locant).
    """

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
    if attachment_atom is not None:
        feature_atoms.add(attachment_atom)
    for group in perceived_groups:
        if group.attachment_carbon in parent_atoms:
            feature_atoms.add(group.attachment_carbon)

    def gated(allow_nonaromatic: bool, allow_relocated_indicated_h: bool = False) -> list[RetainedFusedTemplateMatch]:
        return [
            match
            for match in match_retained_fused_templates(
                mol,
                parent_atoms,
                include_disabled=True,
                allow_nonaromatic=allow_nonaromatic,
                allow_relocated_indicated_h=allow_relocated_indicated_h,
            )
            if match.template.name in PRODUCTION_RETAINED_FUSED_PARENTS
            and match.template.derivative_production_enabled
            and (principal_key != "ketone" or _has_mancude_unsaturation(mol, parent_atoms, match))
        ]

    matches = gated(False) or gated(True) or gated(False, True)
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
    if not _added_hydrogen_is_citable(mol, matches[0]):
        return None
    return ProductionRetainedFusedParent(
        name=parent_name,
        locant_maps=maps,
        metadata=RetainedParentMetadata(
            default_indicated_h=template.default_indicated_h,
            fusion_locants=template.fusion_atoms,
            derivative_stem=template.derivative_stem,
            indicated_hydrogen_count=template.indicated_hydrogen_count,
            mancude_double_bonds=template.mancude_double_bonds or 0,
        ),
    )


def _added_hydrogen_is_citable(mol: Molecule, match: RetainedFusedTemplateMatch) -> bool:
    """Whether every saturated position of the parent can be spelt out.

    A lactam such as 4-oxoquinoline-3-carboxamide saturates two ring positions
    beyond what quinoline supports, and which two cannot be recovered from the
    name -- OPSIN puts the hydrogen on C3 and reads a different molecule.  They
    have to be cited, ``4-oxo-1,4-dihydroquinoline-3-carboxamide``, and only
    parents wired into the indicated-hydrogen machinery can do that.  A parent
    needing an uncitable added hydrogen falls back to von Baeyer, which states
    its saturation positionally and so stays unambiguous.

    Only a position the parent itself holds a mancude bond at can be saturated
    "beyond" it.  Dibenzofuran's oxygen and quinolizine's bridgehead nitrogen
    are single-bonded in the parent hydride, so they are not added hydrogen --
    the template records that as ``aromatic: false``.
    """

    template = match.template
    atom_by_locant = template.atom_by_locant
    mancude_positions = 0
    saturated = 0
    for atom, locant in match.atom_to_locant.items():
        if not atom_by_locant[locant].aromatic:
            continue
        mancude_positions += 1
        ring_bonds = [
            bond
            for neighbor in mol.get_neighbors(atom)
            if neighbor in match.matched_atoms and (bond := mol.get_bond(atom, neighbor)) is not None
        ]
        if ring_bonds and sum(bond.order for bond in ring_bonds) == len(ring_bonds):
            saturated += 1
    # A fully saturated ring system is not a hydro derivative of this parent;
    # it has its own name, and von Baeyer states its saturation positionally.
    if saturated and saturated >= mancude_positions:
        return False
    # Indicated-hydrogen sites the template already marks non-mancude were not
    # counted above, so they must not be subtracted again.
    already_excluded = sum(1 for locant in template.default_indicated_h if not atom_by_locant[locant].aromatic)
    surplus = saturated - (template.indicated_hydrogen_count - already_excluded)
    if surplus <= 0:
        return True
    # A hydro prefix saturates whole double bonds, so an even surplus is citable
    # as one: 1,2,3,4-tetrahydroquinoline, 4-oxo-1,4-dihydroquinoline-3-carboxamide.
    if surplus % 2 == 0:
        return True
    return template.name in INDICATED_H_RETAINED_NAMES


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
