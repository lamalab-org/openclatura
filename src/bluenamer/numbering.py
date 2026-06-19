"""Parent numbering selection."""

import re
from dataclasses import dataclass

from .locants import get_atom_locants, get_bond_locants, parse_locant
from .molecule import Molecule
from .namer_config import INDICATED_H_RETAINED_NAMES
from .naming_data import mapping

NUMBERING_CRITERIA = mapping("numbering_criteria")


@dataclass(frozen=True)
class NumberingPreference:
    """Comparable data used to choose one candidate parent numbering.

    The fields mirror the orientation rule shape used by `src/openiupac`: a
    candidate is scored from structural locants first, then parent suffix and
    substituent locants.  Keeping the tuple named makes future Blue Book rule
    additions local to this layer instead of scattered through descriptor
    builders.
    """

    principal: tuple[int, ...]
    hetero_by_priority: tuple[tuple[int, ...], ...]
    indicated_hydrogen: tuple[int, ...]
    unsaturation: tuple[int, ...]
    substituent_and_unsaturation: tuple[int, ...]
    substituent_citation: tuple[int, ...]
    stereochemistry: tuple[int, ...]

    def criterion_value(self, criterion: str) -> tuple:
        if criterion == "principal":
            return self.principal
        if criterion == "hetero_by_priority":
            return self.hetero_by_priority
        if criterion == "indicated_hydrogen":
            return self.indicated_hydrogen
        if criterion == "unsaturation":
            return self.unsaturation
        if criterion == "substituent_and_unsaturation":
            return self.substituent_and_unsaturation
        if criterion == "substituent_citation":
            return self.substituent_citation
        if criterion == "stereochemistry":
            return self.stereochemistry
        raise KeyError(f"Unknown numbering criterion: {criterion}")

    def ordered_key(self, criteria: list[str]) -> tuple:
        return tuple(self.criterion_value(criterion) for criterion in criteria)

    def chain_key(self) -> tuple:
        return self.ordered_key(NUMBERING_CRITERIA["chain"])

    def ring_key(self) -> tuple:
        return self.ordered_key(NUMBERING_CRITERIA["ring"])


def polycycle_numbering_key(
    mol: Molecule,
    numbered_path: list[int],
    *,
    include_saturated_ring_proxy: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the openiupac-style orientation key for ring frameworks.

    `src/openiupac.parent_selection._polycycle_numbering_key` compares
    heteroatom locants, then indicated-hydrogen locants.  This adapter uses the
    active lightweight `Molecule` model; explicit hydrogens/aromaticity are not
    stored here, so the saturated-ring proxy is opt-in until callers can prove
    it is appropriate for the parent class being numbered.
    """

    return (
        _heteroatom_locants_by_priority(mol, numbered_path),
        _indicated_hydrogen_like_locants(
            mol,
            numbered_path,
            retained_name=None,
            include_all_ring_carbons=include_saturated_ring_proxy,
        ),
    )


def number_parent(
    mol: Molecule,
    candidate_paths: list[list[int]],
    principal_carbons: set[int],
    substituent_mapping: dict[int, list[str]],
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool = False,
    fixed_start: bool = False,
    retained_name: str = None,
) -> list[int]:
    """Choose the preferred numbering for a selected parent skeleton."""

    candidates = []
    if is_bicycle or is_spiro or is_polycycle or fixed_start:
        candidates = candidate_paths
    elif is_ring:
        path = candidate_paths[0]
        for i in range(len(path)):
            rotated = path[i:] + path[:i]
            candidates.append(rotated)
            candidates.append(rotated[::-1])
    else:
        path = candidate_paths[0]
        candidates.append(path)
        candidates.append(path[::-1])

    def compare_paths(p1, p2):
        def evaluate(oriented_path):
            preference = _numbering_preference(
                mol,
                oriented_path,
                principal_carbons,
                substituent_mapping,
                is_bicycle=is_bicycle,
                is_spiro=is_spiro,
                is_polycycle=is_polycycle,
                retained_name=retained_name,
            )
            if is_ring:
                return preference.ring_key()
            return preference.chain_key()

        ev1 = evaluate(p1)
        ev2 = evaluate(p2)

        for v1, v2 in zip(ev1, ev2):
            if not v1 and not v2:
                continue
            if not v1:
                return 1
            if not v2:
                return -1
            for x, y in zip(v1, v2):
                if x < y:
                    return -1
                if x > y:
                    return 1
            if len(v1) < len(v2):
                return -1
            if len(v1) > len(v2):
                return 1
        return 0

    best = candidates[0]
    for c in candidates[1:]:
        if compare_paths(c, best) < 0:
            best = c
    return best


def choose_parent_numbering(
    mol: Molecule,
    candidate_paths: list[list[int]],
    principal_atoms,
    substituent_mapping: dict[int, list],
    locant_maps,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    retained_name: str | None,
    *,
    fixed_start: bool = False,
) -> tuple[list[int], dict[int, str] | None]:
    """Choose parent numbering from retained locant maps or normal rules."""

    principal_atom_set = set(principal_atoms)
    if locant_maps:

        def evaluate_map(lmap):
            def get_val(idx):
                return parse_locant(lmap[idx])

            principal_eval = sorted([get_val(idx) for idx in principal_atom_set if idx in lmap])
            het_by_priority = {}
            for atom in mol:
                if atom.idx in lmap and not atom.is_carbon:
                    priority = atom.element.hw_priority or 99
                    het_by_priority.setdefault(priority, []).append(atom.idx)
            heteroatom_eval = tuple(
                sorted([get_val(idx) for idx in het_by_priority[priority]])
                for priority in sorted(het_by_priority.keys())
            )
            substituent_eval = sorted([get_val(idx) for idx in set(substituent_mapping.keys()) if idx in lmap])
            indicated_h_eval = sorted(
                get_val(idx)
                for idx in _indicated_hydrogen_like_atoms(
                    mol,
                    [idx for idx in lmap if idx in mol.atoms],
                    retained_name=retained_name,
                    include_all_ring_carbons=False,
                )
                if idx in lmap
            )
            return heteroatom_eval + (tuple(indicated_h_eval), principal_eval, substituent_eval)

        locant_map = min(locant_maps, key=evaluate_map)
        return list(locant_map.keys()), locant_map

    numbered_path = number_parent(
        mol,
        candidate_paths,
        principal_atom_set,
        substituent_mapping,
        is_ring,
        is_bicycle,
        is_spiro,
        is_polycycle=is_polycycle,
        fixed_start=fixed_start,
        retained_name=retained_name,
    )
    if is_bicycle or is_spiro or is_polycycle:
        return numbered_path, {atom_idx: str(locant) for locant, atom_idx in enumerate(numbered_path, start=1)}
    return numbered_path, None


def _numbering_preference(
    mol: Molecule,
    oriented_path: list[int],
    principal_carbons: set[int],
    substituent_mapping: dict[int, list[str]],
    *,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    retained_name: str | None,
) -> NumberingPreference:
    principal = tuple(get_atom_locants(oriented_path, principal_carbons))
    hetero_by_priority = _heteroatom_locants_by_priority(mol, oriented_path)
    indicated_hydrogen = _indicated_hydrogen_like_locants(
        mol,
        oriented_path,
        retained_name=retained_name,
        include_all_ring_carbons=False,
    )
    substituent_locants = get_atom_locants(oriented_path, set(substituent_mapping.keys()))
    double_bonds, triple_bonds = get_bond_locants(mol, oriented_path, is_bicycle, is_spiro, is_polycycle)
    unsaturation = () if retained_name else tuple(sorted(double_bonds + triple_bonds))
    substituent_and_unsaturation = tuple(sorted(substituent_locants + list(unsaturation)))
    return NumberingPreference(
        principal=principal,
        hetero_by_priority=hetero_by_priority,
        indicated_hydrogen=indicated_hydrogen,
        unsaturation=unsaturation,
        substituent_and_unsaturation=substituent_and_unsaturation,
        substituent_citation=_substituent_citation_locants(oriented_path, substituent_mapping),
        stereochemistry=_stereochemistry_sequence(mol, oriented_path),
    )


def _heteroatom_locants_by_priority(mol: Molecule, oriented_path: list[int]) -> tuple[tuple[int, ...], ...]:
    hetero_by_priority: dict[int, list[int]] = {}
    parent_atoms = set(oriented_path)
    for atom in mol:
        if atom.idx in parent_atoms and not atom.is_carbon:
            priority = atom.element.hw_priority or 99
            hetero_by_priority.setdefault(priority, []).append(atom.idx)
    return tuple(
        tuple(get_atom_locants(oriented_path, set(hetero_by_priority[priority])))
        for priority in sorted(hetero_by_priority)
    )


def _indicated_hydrogen_like_locants(
    mol: Molecule,
    oriented_path: list[int],
    *,
    retained_name: str | None,
    include_all_ring_carbons: bool,
) -> tuple[int, ...]:
    atoms = _indicated_hydrogen_like_atoms(
        mol,
        oriented_path,
        retained_name=retained_name,
        include_all_ring_carbons=include_all_ring_carbons,
    )
    return tuple(get_atom_locants(oriented_path, atoms))


def _indicated_hydrogen_like_atoms(
    mol: Molecule,
    oriented_path: list[int],
    *,
    retained_name: str | None,
    include_all_ring_carbons: bool,
) -> set[int]:
    charged_tetrazole = retained_name == "tetrazole" and any(mol.atoms[a_idx].charge for a_idx in oriented_path)
    if retained_name not in INDICATED_H_RETAINED_NAMES and not include_all_ring_carbons:
        return set()
    if charged_tetrazole:
        return set()
    parent_atoms = set(oriented_path)
    saturated_atoms = set()
    for atom_idx in oriented_path:
        if include_all_ring_carbons:
            if _has_indicated_hydrogen_metadata(mol, atom_idx, parent_atoms):
                saturated_atoms.add(atom_idx)
        elif _has_retained_indicated_hydrogen_proxy(mol, atom_idx, parent_atoms):
            saturated_atoms.add(atom_idx)
    return saturated_atoms


def _has_indicated_hydrogen_metadata(
    mol: Molecule,
    atom_idx: int,
    parent_atoms: set[int],
) -> bool:
    atom = mol.atoms[atom_idx]
    ring_bonds = [
        mol.get_bond(atom_idx, neighbor) for neighbor in mol.get_neighbors(atom_idx) if neighbor in parent_atoms
    ]
    is_ring_atom = len(ring_bonds) >= 2
    if not is_ring_atom or atom.total_h_count <= 0:
        return False
    return (atom.is_aromatic and not atom.is_carbon) or (atom.is_carbon and not atom.is_aromatic)


def _has_retained_indicated_hydrogen_proxy(mol: Molecule, atom_idx: int, parent_atoms: set[int]) -> bool:
    atom = mol.atoms[atom_idx]
    if atom.symbol not in {"C", "N"}:
        return False
    ring_bonds = [
        mol.get_bond(atom_idx, neighbor) for neighbor in mol.get_neighbors(atom_idx) if neighbor in parent_atoms
    ]
    return sum(bond.order for bond in ring_bonds if bond is not None) == 2


def _substituent_citation_locants(
    oriented_path: list[int], substituent_mapping: dict[int, list[str]]
) -> tuple[int, ...]:
    alpha_list = []
    for idx in oriented_path:
        if idx not in substituent_mapping:
            continue
        locant = oriented_path.index(idx) + 1
        for item in substituent_mapping[idx]:
            name = item.name if hasattr(item, "name") else item
            alpha_list.append((_substituent_sort_key(name), locant))
    alpha_list.sort(key=lambda item: item[0])
    return tuple(locant for _, locant in alpha_list)


def _substituent_sort_key(name: str) -> str:
    normalized = name.lower()
    normalized = re.sub(r"^[\(\[\{\)]+", "", normalized)
    prefix_pattern = r"^((?:(?:[0-9]+[a-z]*|[nospmc]\'*)(?:,(?:[0-9]+[a-z]*|[nospmc]\'*))*|[ezrs]+|sec|tert|t|s|d|l|m|o|p|alpha|beta|gamma))([-)]+)"
    while True:
        match = re.match(prefix_pattern, normalized)
        if not match:
            break
        normalized = normalized[match.end() :]
        normalized = re.sub(r"^[\(\[\{\)]+", "", normalized)
    return normalized


def _stereochemistry_sequence(mol: Molecule, oriented_path: list[int]) -> tuple[int, ...]:
    sequence = []
    for atom_idx in oriented_path:
        stereo = mol.atoms[atom_idx].stereo
        if stereo:
            sequence.append(0 if stereo == "R" else 1)
    return tuple(sequence)
