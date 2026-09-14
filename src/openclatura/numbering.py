"""Parent numbering selection."""

import re
from dataclasses import dataclass

from .locants import get_atom_locants, get_bond_locants, parse_locant
from .molecule import Molecule
from .namer_config import INDICATED_H_ELEMENTS, cites_indicated_hydrogen
from .naming_data import mapping
from .rules.retained import mancude_monocycle_hydro_plan

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
    hydro: tuple[int, ...]
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
        if criterion == "hydro":
            return self.hydro
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


class _DeferredNumberingKey:
    """The ordered criteria, each derived the first time it is compared.

    A comparison settles on the first criterion that differs, and over the
    corpus 96% of them settle within three of the eight. Materialising the
    whole key derived hydro sites, bond locants, substituent citations and a
    stereochemistry walk that nothing went on to read.
    """

    __slots__ = ("_criteria", "_preference", "_values")

    def __init__(self, preference: "_DeferredNumberingPreference", criteria: list[str]) -> None:
        self._preference = preference
        self._criteria = criteria
        self._values: list[tuple] = []

    def __len__(self) -> int:
        return len(self._criteria)

    def __getitem__(self, index: int) -> tuple:
        values = self._values
        while len(values) <= index:
            values.append(self._preference.criterion_value(self._criteria[len(values)]))
        return values[index]

    def __iter__(self):
        for index in range(len(self._criteria)):
            yield self[index]


class _DeferredNumberingPreference:
    """A NumberingPreference whose criteria are derived on demand.

    It holds what a criterion is derived from rather than the criteria, and
    answers criterion_value with the same values the eager dataclass carries.
    Each is derived once per candidate however often it is compared.
    """

    __slots__ = (
        "_mol",
        "_oriented_path",
        "_principal_carbons",
        "_substituent_mapping",
        "_is_bicycle",
        "_is_spiro",
        "_is_polycycle",
        "_retained_name",
        "_values",
    )

    def __init__(
        self,
        mol: Molecule,
        oriented_path: list[int],
        principal_carbons: set[int],
        substituent_mapping: dict[int, list[str]],
        *,
        is_bicycle: bool,
        is_spiro: bool,
        is_polycycle: bool,
        retained_name: str | None,
    ) -> None:
        self._mol = mol
        self._oriented_path = oriented_path
        self._principal_carbons = principal_carbons
        self._substituent_mapping = substituent_mapping
        self._is_bicycle = is_bicycle
        self._is_spiro = is_spiro
        self._is_polycycle = is_polycycle
        self._retained_name = retained_name
        self._values: dict[str, tuple] = {}

    def _unsaturation(self) -> tuple[int, ...]:
        if self._retained_name:
            return ()
        double_bonds, triple_bonds = get_bond_locants(
            self._mol, self._oriented_path, self._is_bicycle, self._is_spiro, self._is_polycycle
        )
        return tuple(sorted(double_bonds + triple_bonds))

    def _derive(self, criterion: str) -> tuple:
        if criterion == "principal":
            return tuple(get_atom_locants(self._oriented_path, self._principal_carbons))
        if criterion == "hetero_by_priority":
            return _heteroatom_locants_by_priority(
                self._mol,
                self._oriented_path,
                monocycle=not (self._is_bicycle or self._is_spiro or self._is_polycycle),
            )
        if criterion == "indicated_hydrogen":
            return _indicated_hydrogen_like_locants(
                self._mol,
                self._oriented_path,
                retained_name=self._retained_name,
                include_all_ring_carbons=False,
            )
        if criterion == "hydro":
            return _hydro_locants(self._mol, self._oriented_path, self._retained_name)
        if criterion == "unsaturation":
            return self._unsaturation()
        if criterion == "substituent_and_unsaturation":
            substituent_locants = get_atom_locants(self._oriented_path, set(self._substituent_mapping.keys()))
            return tuple(sorted(substituent_locants + list(self.criterion_value("unsaturation"))))
        if criterion == "substituent_citation":
            return _substituent_citation_locants(self._oriented_path, self._substituent_mapping)
        if criterion == "stereochemistry":
            return _stereochemistry_sequence(self._mol, self._oriented_path)
        raise KeyError(f"Unknown numbering criterion: {criterion}")

    def criterion_value(self, criterion: str) -> tuple:
        values = self._values
        if criterion not in values:
            values[criterion] = self._derive(criterion)
        return values[criterion]

    def ordered_key(self, criteria: list[str]) -> "_DeferredNumberingKey":
        return _DeferredNumberingKey(self, criteria)

    def chain_key(self) -> "_DeferredNumberingKey":
        return self.ordered_key(NUMBERING_CRITERIA["chain"])

    def ring_key(self) -> "_DeferredNumberingKey":
        return self.ordered_key(NUMBERING_CRITERIA["ring"])


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

    def evaluate(oriented_path):
        preference = _DeferredNumberingPreference(
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

    def compare_evaluations(ev1, ev2):
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

    # Each candidate is scored once. Comparing paths rather than scores made
    # the running best pay for a fresh evaluation on every comparison, which
    # was half of all the scoring this function did.
    best = candidates[0]
    best_evaluation = evaluate(best)
    for c in candidates[1:]:
        evaluation = evaluate(c)
        if compare_evaluations(evaluation, best_evaluation) < 0:
            best = c
            best_evaluation = evaluation
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
    proven_hydrogen_locants: dict[frozenset[tuple[int, str]], tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> tuple[list[int], dict[int, str] | None]:
    """Choose parent numbering from retained locant maps or normal rules."""

    principal_atom_set = set(principal_atoms)
    if locant_maps:
        if len(locant_maps) == 1:
            locant_map = locant_maps[0]
            return list(locant_map.keys()), locant_map

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
            # Low locants to hydro prefixes, P-31.1.4.2.4.
            hydro_eval = sorted(
                get_val(idx) for idx in lmap if idx in mol.atoms and _is_saturated_ring_site(mol, idx, list(lmap))
            )
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
            if proven_hydrogen_locants is not None:
                indicated, hydro = proven_hydrogen_locants[frozenset(lmap.items())]
                indicated_h_eval = sorted(parse_locant(locant) for locant in indicated)
                hydro_eval = sorted(parse_locant(locant) for locant in hydro)
            return heteroatom_eval + (tuple(indicated_h_eval), principal_eval, hydro_eval, substituent_eval)

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
    hetero_by_priority = _heteroatom_locants_by_priority(
        mol,
        oriented_path,
        monocycle=not (is_bicycle or is_spiro or is_polycycle),
    )
    indicated_hydrogen = _indicated_hydrogen_like_locants(
        mol,
        oriented_path,
        retained_name=retained_name,
        include_all_ring_carbons=False,
    )
    hydro = _hydro_locants(mol, oriented_path, retained_name)
    substituent_locants = get_atom_locants(oriented_path, set(substituent_mapping.keys()))
    double_bonds, triple_bonds = get_bond_locants(mol, oriented_path, is_bicycle, is_spiro, is_polycycle)
    unsaturation = () if retained_name else tuple(sorted(double_bonds + triple_bonds))
    substituent_and_unsaturation = tuple(sorted(substituent_locants + list(unsaturation)))
    return NumberingPreference(
        principal=principal,
        hetero_by_priority=hetero_by_priority,
        indicated_hydrogen=indicated_hydrogen,
        hydro=hydro,
        unsaturation=unsaturation,
        substituent_and_unsaturation=substituent_and_unsaturation,
        substituent_citation=_substituent_citation_locants(oriented_path, substituent_mapping),
        stereochemistry=_stereochemistry_sequence(mol, oriented_path),
    )


def _hydro_locants(mol: Molecule, oriented_path: list[int], retained_name: str | None) -> tuple[int, ...]:
    """Locants a hydro derivative of a retained mancude parent would cite."""

    split = _mancude_hydro_split(mol, oriented_path, retained_name)
    return () if split is None else tuple(get_atom_locants(oriented_path, split[1]))


def _mancude_hydro_split(
    mol: Molecule, oriented_path: list[int], retained_name: str | None
) -> tuple[set[int], set[int]] | None:
    """Split a hydro derivative's saturated positions into cited H and hydro."""

    plan = mancude_monocycle_hydro_plan(mol, oriented_path, retained_name)
    if plan is None:
        return None
    indicated, _, citable = plan
    citable = sorted(citable, key=oriented_path.index)
    return set(citable[:indicated]), set(citable[indicated:])


def _heteroatom_locants_by_priority(
    mol: Molecule,
    oriented_path: list[int],
    *,
    monocycle: bool = False,
) -> tuple[tuple[int, ...], ...]:
    hetero_by_priority: dict[int, list[int]] = {}
    parent_atoms = set(oriented_path)
    for atom in mol:
        if atom.idx in parent_atoms and not atom.is_carbon:
            priority = atom.element.hw_priority or 99
            hetero_by_priority.setdefault(priority, []).append(atom.idx)
    by_priority = tuple(
        tuple(get_atom_locants(oriented_path, set(hetero_by_priority[priority])))
        for priority in sorted(hetero_by_priority)
    )
    if not monocycle:
        return by_priority
    real = {
        priority: [idx for idx in group if idx not in mol.substituted_symbols]
        for priority, group in hetero_by_priority.items()
    }
    key = _monocycle_heteroatom_key({priority: group for priority, group in real.items() if group}, oriented_path)
    return (*key, *by_priority)


def _monocycle_heteroatom_key(
    hetero_by_priority: dict[int, list[int]],
    oriented_path: list[int],
) -> tuple[tuple[int, ...], ...]:
    """The two criteria a Hantzsch-Widman ring settles before seniority order.

    Locant 1 goes to the most senior heteroatom (S in 1,3,4-thiadiazole), then
    the heteroatoms as a set take the lowest locants (1,4,2-dioxazole).
    """

    locants = {
        locant: priority
        for priority, group in hetero_by_priority.items()
        for locant in get_atom_locants(oriented_path, set(group))
    }
    if not locants:
        return ((), ())
    return ((locants[min(locants)],), tuple(sorted(locants)))


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
    split = _mancude_hydro_split(mol, oriented_path, retained_name)
    if split is not None:
        return split[0]
    charged_tetrazole = retained_name == "tetrazole" and any(mol.atoms[a_idx].charge for a_idx in oriented_path)
    if not cites_indicated_hydrogen(retained_name) and not include_all_ring_carbons:
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
    if atom.symbol not in INDICATED_H_ELEMENTS:
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


def _is_saturated_ring_site(mol: Molecule, atom_idx: int, ring_atoms: list[int]) -> bool:
    """A ring position a hydro prefix would cite, read off the structure.

    ``_hydro_locants`` answers the same question from the monocycle hydro plan,
    which a fused parent has none of.
    """

    ring_bonds = [bond for n in mol.get_neighbors(atom_idx) if n in ring_atoms and (bond := mol.get_bond(atom_idx, n))]
    # Substitution does not un-saturate a position, so H count is not consulted:
    # indane's C1 is a hydro site whether or not it carries the diazo group.
    return bool(ring_bonds) and all(bond.order == 1 for bond in ring_bonds)
