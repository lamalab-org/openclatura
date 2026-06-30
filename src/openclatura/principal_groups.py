"""Principal characteristic-group selection and suffix assembly."""

from .assembly_parts import AssemblyParts, PrincipalGroupItem
from .group_atom_roles import hydrazone_characteristic_carbon
from .locants import parse_locant
from .molecule import Molecule
from .nomenclature import RULES
from .perception import PerceivedGroup, perceive_groups
from .trace_helpers import bond_ids_within


def component_groups(mol: Molecule, component_atoms: set[int]) -> list[PerceivedGroup]:
    """Return perceived groups whose attachment atom is inside a component."""

    return [group for group in perceive_groups(mol) if group.attachment_carbon in component_atoms]


def component_principal_key(perceived_groups: list[PerceivedGroup], is_substituent: bool) -> str | None:
    """Select the senior principal characteristic group for a component."""

    if is_substituent:
        return None
    candidates = [group.key for group in perceived_groups if group.is_principal_candidate]
    return RULES.functional_groups.most_senior(candidates).key if candidates else None


def partition_principal_and_prefix_groups(
    perceived_groups: list[PerceivedGroup], principal_key: str | None
) -> tuple[list[int], list[PerceivedGroup]]:
    """Split perceived groups into principal attachment atoms and prefixes."""

    principal_carbons = []
    prefix_groups = []
    for group in perceived_groups:
        if group.key == principal_key:
            principal_carbons.append(group.attachment_carbon)
        else:
            prefix_groups.append(group)
    return principal_carbons, prefix_groups


def filter_component_groups_to_parent(
    perceived_groups: list[PerceivedGroup], parent_set: set[int], is_substituent: bool
) -> tuple[list[PerceivedGroup], str | None, list[int], list[PerceivedGroup]]:
    """Keep only groups attached to the selected parent and recompute seniority."""

    valid_groups = [group for group in perceived_groups if group.attachment_carbon in parent_set]
    principal_key = component_principal_key(valid_groups, is_substituent)
    principal_carbons, prefix_groups = partition_principal_and_prefix_groups(valid_groups, principal_key)
    return valid_groups, principal_key, principal_carbons, prefix_groups


def add_component_principal_group(
    mol: Molecule,
    parts: AssemblyParts,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    principal_carbons: list[int],
    numbered_path: list[int],
    get_loc,
) -> None:
    """Add the principal characteristic group suffix locants to assembly parts."""

    if not principal_key:
        return
    locants = sorted([get_loc(c) for c in principal_carbons if c in numbered_path], key=parse_locant)
    atom_ids = set()
    for group in perceived_groups:
        if group.key == principal_key and group.attachment_carbon in numbered_path:
            atom_ids.add(group.attachment_carbon)
            atom_ids.update(group.atoms_involved)
            if group.key in RULES.functional_groups.keys_with_family("hydrazone"):
                hydrazone_carbon = hydrazone_characteristic_carbon(mol, group)
                if hydrazone_carbon is not None:
                    atom_ids.add(hydrazone_carbon)
                    if _hydrazone_allows_unlocanted_stereo(mol, group, hydrazone_carbon):
                        for nitrogen in [idx for idx in group.atoms_involved if mol.atoms[idx].symbol == "N"]:
                            bond = mol.get_bond(hydrazone_carbon, nitrogen)
                            if bond is not None and bond.order == 2 and bond.stereo in {"E", "Z"}:
                                parts.stereo_features.append(("", bond.stereo))
    parts.principal_group = PrincipalGroupItem(
        key=principal_key,
        locants=locants,
        atom_ids=atom_ids,
        bond_ids=bond_ids_within(mol, atom_ids),
        charge_atom_ids={atom_idx for atom_idx in atom_ids if mol.atoms[atom_idx].charge != 0},
    )


def _hydrazone_allows_unlocanted_stereo(mol: Molecule, group: PerceivedGroup, carbon: int) -> bool:
    """Return whether hydrazone E/Z can be rendered as a parent-name prefix.

    OPSIN accepts unlocanted `(E)`/`(Z)` for ordinary aldehyde hydrazones such
    as benzaldehyde hydrazone. It does not reliably attach that descriptor when
    the hydrazone N-N side is itself part of an azo/diazo/chalcogen-imide chain.
    This guard keeps stereo rendering tied to the simple hydrazone suffix class
    instead of broadcasting it across every C=N-N fragment.
    """

    if group.key not in {"aldehyde_hydrazone", "ring_aldehyde_hydrazone"}:
        return False
    hydrazone_n = None
    terminal_n = None
    for atom_idx in group.atoms_involved:
        if mol.atoms[atom_idx].symbol != "N":
            continue
        bond = mol.get_bond(carbon, atom_idx)
        if bond is not None and bond.order == 2:
            hydrazone_n = atom_idx
            break
    if hydrazone_n is None:
        return False
    carbon_substituents = [n for n in mol.get_neighbors(carbon) if n != hydrazone_n]
    if len(carbon_substituents) != 1 or mol.atoms[carbon_substituents[0]].symbol != "C":
        return False
    terminal_candidates = [
        n
        for n in mol.get_neighbors(hydrazone_n)
        if n != carbon and mol.atoms[n].symbol == "N" and n in group.atoms_involved
    ]
    if len(terminal_candidates) != 1:
        return False
    terminal_n = terminal_candidates[0]
    for neighbor in mol.get_neighbors(terminal_n):
        if neighbor == hydrazone_n:
            continue
        bond = mol.get_bond(terminal_n, neighbor)
        if bond is not None and bond.order != 1:
            return False
        if mol.atoms[neighbor].symbol not in {"C", "H"}:
            return False
    return True
