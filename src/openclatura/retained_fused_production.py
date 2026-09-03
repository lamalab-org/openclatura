from dataclasses import dataclass

from .assembly_parts import RetainedParentMetadata, SubstituentItem
from .grammar_snapshot_data import retained_fused_derivative_gate
from .molecule import Molecule
from .namer_config import INDICATED_H_RETAINED_NAMES
from .perception import PerceivedGroup
from .retained_fused_templates import RetainedGraphTemplateMatch, match_retained_fused_templates
from .retained_name_policy import retained_parent_name_policy

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

    def eligible(matches: list[RetainedGraphTemplateMatch]) -> list[RetainedGraphTemplateMatch]:
        # A match that places the indicated hydrogen spells its saturation; prefer it.
        matches = sorted(matches, key=lambda match: not match.indicated_h)
        return [
            match
            for match in matches
            if match.template.name in PRODUCTION_RETAINED_FUSED_PARENTS
            and match.template.derivative_production_enabled
            and (
                principal_key != "ketone"
                or _has_mancude_unsaturation(mol, parent_atoms, match)
                or _saturation_is_hydro_citable(mol, parent_atoms, match)
            )
            # A match whose saturation cannot be spelt out must not block a later
            # (relocated indicated-H) match that can: 1,3-dihydro-2H-1,4-benzodiazepin-2-one.
            and _added_hydrogen_is_citable(mol, match)
        ]

    strict_matches = match_retained_fused_templates(mol, parent_atoms)
    matches = eligible(strict_matches)
    if strict_matches and not matches:
        # A strict enabled parent has already supplied an exact locant map.
        # Do not replace it with a weaker hydro/tautomer match merely because
        # the later derivative-override vocabulary has not adopted its token.
        return None
    if not matches:
        # Pool the derivative modes: each atom map is kept under the first mode that can
        # spell it, a map that places the indicated hydrogen is preferred for the parent
        # (1,3-dihydro-2H-…-2-one over 1H-…-4-one), and only maps sharing that placement
        # stay for the numbering layer.
        pooled: dict[tuple[tuple[int, str], ...], RetainedGraphTemplateMatch] = {}
        for kwargs in (
            {"allow_nonaromatic": True},
            {"allow_relocated_indicated_h": True},
            {"allow_nonaromatic": True, "allow_relocated_indicated_h": True},
        ):
            for match in eligible(match_retained_fused_templates(mol, parent_atoms, **kwargs)):
                pooled.setdefault((match.template.name, tuple(sorted(match.atom_to_locant.items()))), match)
        if pooled:
            ordered = list(pooled.values())
            best = next((m for m in ordered if m.indicated_h), ordered[0])
            matches = [
                m for m in ordered if m.template.name == best.template.name and m.indicated_h == best.indicated_h
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
    if not _added_hydrogen_is_citable(mol, matches[0]):
        return None
    output_context = (
        "unsubstituted_parent"
        if attachment_atom is None and not substituent_mapping and principal_key is None
        else "composite_parent"
    )
    name_policy = retained_parent_name_policy(template.name)
    output_name = name_policy.output_name(output_context) if name_policy is not None else template.name
    return ProductionRetainedFusedParent(
        name=output_name,
        locant_maps=maps,
        metadata=RetainedParentMetadata(
            default_indicated_h=matches[0].indicated_h,
            fusion_locants=template.fusion_atoms,
            derivative_stem=template.derivative_stem,
            indicated_hydrogen_count=template.indicated_hydrogen_count,
            mancude_double_bonds=template.mancude_double_bonds or 0,
            relocated_indicated_h=matches[0].indicated_h != template.default_indicated_h,
        ),
    )


def _added_hydrogen_is_citable(mol: Molecule, match: RetainedGraphTemplateMatch) -> bool:
    """
    Whether every saturated position of the parent can be spelt out.
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

    if saturated and saturated >= mancude_positions:
        return False
    already_excluded = sum(1 for locant in match.indicated_h if not atom_by_locant[locant].aromatic)
    surplus = saturated - (template.indicated_hydrogen_count - already_excluded)
    if surplus <= 0:
        return True
    if surplus % 2 == 0:
        return True
    return template.name in INDICATED_H_RETAINED_NAMES


def _neutral_component(mol: Molecule, atoms: set[int]) -> bool:
    return all(mol.atoms[atom].charge == 0 for atom in atoms)


def _has_mancude_unsaturation(
    mol: Molecule,
    parent_atoms: set[int],
    match: RetainedGraphTemplateMatch,
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


def _saturation_is_hydro_citable(mol: Molecule, parent_atoms: set[int], match: RetainedGraphTemplateMatch) -> bool:
    """A ketone on a partly saturated mancude parent is spelt with hydro prefixes and
    added hydrogen (3,4-dihydroquinolin-2(1H)-one) when the saturated positions form an
    even count once the ketone carbon and the parent's own indicated hydrogen are taken out."""

    template = match.template
    if template.mancude_double_bonds is None:
        return False
    saturated = 0
    for atom, locant in match.atom_to_locant.items():
        if not template.atom_by_locant[locant].aromatic:
            continue
        ring_bonds = [
            bond
            for neighbor in mol.get_neighbors(atom)
            if neighbor in match.matched_atoms and (bond := mol.get_bond(atom, neighbor)) is not None
        ]
        if ring_bonds and all(bond.order == 1 for bond in ring_bonds):
            saturated += 1
    return saturated > 0 and saturated < len(match.atom_to_locant)


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
