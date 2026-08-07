"""Graph templates for retained monocyclic parent hydrides.

Templates describe cyclic graphs and their locants.  They are indexed by a
cheap graph signature, then matched by enumerating the two orientations of the
cycle.  No SMILES or SMARTS strings participate in production matching.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .molecule import Molecule
from .naming_data import load_json_table


@dataclass(frozen=True)
class RetainedMonocycleAtomTemplate:
    locant: str
    symbol: str = "C"
    charge: int = 0


@dataclass(frozen=True)
class RetainedMonocycleGraphTemplate:
    name: str
    priority: int
    locants: tuple[str, ...]
    atoms: tuple[RetainedMonocycleAtomTemplate, ...]
    double_bond_locants: tuple[tuple[str, str], ...]
    no_cumulated_double_bonds: bool = True

    @property
    def atom_by_locant(self) -> dict[str, RetainedMonocycleAtomTemplate]:
        return {atom.locant: atom for atom in self.atoms}

    @property
    def signature(self) -> tuple[int, tuple[tuple[str, int], ...], int]:
        return (
            len(self.locants),
            tuple(sorted(Counter(atom.symbol for atom in self.atoms).items())),
            len(self.double_bond_locants),
        )


@dataclass(frozen=True)
class RetainedMonocycleTemplateMatch:
    template: RetainedMonocycleGraphTemplate
    atom_to_locant_maps: tuple[dict[int, str], ...]


@lru_cache(maxsize=1)
def retained_monocycle_graph_templates() -> tuple[RetainedMonocycleGraphTemplate, ...]:
    rows = load_json_table("retained_monocycle_graph_templates.json").get("parents", ())
    templates = tuple(_template_from_data(row) for row in rows)
    for template in templates:
        _validate_template(template)
    return templates


@lru_cache(maxsize=1)
def retained_monocycle_graph_template_names() -> frozenset[str]:
    """Names whose element support is proven by a graph template."""

    return frozenset(template.name for template in retained_monocycle_graph_templates())


@lru_cache(maxsize=1)
def _templates_by_signature() -> dict[tuple[int, tuple[tuple[str, int], ...], int], tuple[RetainedMonocycleGraphTemplate, ...]]:
    indexed: dict[tuple[int, tuple[tuple[str, int], ...], int], list[RetainedMonocycleGraphTemplate]] = {}
    for template in retained_monocycle_graph_templates():
        indexed.setdefault(template.signature, []).append(template)
    return {
        signature: tuple(sorted(templates, key=lambda template: (template.priority, template.name)))
        for signature, templates in indexed.items()
    }


def match_retained_monocycle_templates(
    mol: Molecule,
    atom_indices: set[int] | list[int] | tuple[int, ...],
) -> tuple[RetainedMonocycleTemplateMatch, ...]:
    """Return retained monocycle matches with every valid locant map."""

    atoms = set(atom_indices)
    cycle = _ordered_cycle(mol, atoms)
    if cycle is None:
        return ()
    double_bonds = _double_bond_count(mol, atoms)
    signature = (
        len(atoms),
        tuple(sorted(Counter(mol.atoms[atom].symbol for atom in atoms).items())),
        double_bonds,
    )
    matches = []
    for template in _templates_by_signature().get(signature, ()):
        if template.no_cumulated_double_bonds and _has_cumulated_double_bond(mol, atoms):
            continue
        maps = _matching_locant_maps(mol, cycle, template)
        if maps:
            matches.append(RetainedMonocycleTemplateMatch(template=template, atom_to_locant_maps=maps))
    return tuple(matches)


def _template_from_data(row: dict) -> RetainedMonocycleGraphTemplate:
    locants = tuple(str(locant) for locant in row["locants"])
    heteroatoms = {str(item["locant"]): dict(item) for item in row.get("heteroatoms", ())}
    atoms = []
    for locant in locants:
        item = heteroatoms.get(locant, {})
        atoms.append(
            RetainedMonocycleAtomTemplate(
                locant=locant,
                symbol=str(item.get("symbol", "C")),
                charge=int(item.get("charge", 0)),
            )
        )
    return RetainedMonocycleGraphTemplate(
        name=str(row["name"]),
        priority=int(row.get("priority", 1000)),
        locants=locants,
        atoms=tuple(atoms),
        double_bond_locants=tuple(
            (str(left), str(right)) for left, right in row.get("double_bond_locants", ())
        ),
        no_cumulated_double_bonds=bool(row.get("no_cumulated_double_bonds", True)),
    )


def _validate_template(template: RetainedMonocycleGraphTemplate) -> None:
    if len(template.locants) < 3:
        raise ValueError(f"Retained monocycle {template.name!r} must have at least three locants.")
    if len(set(template.locants)) != len(template.locants):
        raise ValueError(f"Retained monocycle {template.name!r} has duplicate locants.")
    if {atom.locant for atom in template.atoms} != set(template.locants):
        raise ValueError(f"Retained monocycle {template.name!r} atom locants do not match its locant list.")
    cycle_edges = {
        frozenset((template.locants[index], template.locants[(index + 1) % len(template.locants)]))
        for index in range(len(template.locants))
    }
    double_edges = {frozenset(edge) for edge in template.double_bond_locants}
    if len(double_edges) != len(template.double_bond_locants) or not double_edges <= cycle_edges:
        raise ValueError(f"Retained monocycle {template.name!r} has invalid double-bond locants.")


def _ordered_cycle(mol: Molecule, atoms: set[int]) -> tuple[int, ...] | None:
    if len(atoms) < 3:
        return None
    adjacency = {
        atom: tuple(neighbor for neighbor in mol.get_neighbors(atom) if neighbor in atoms)
        for atom in atoms
    }
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None
    start = min(atoms)
    first = min(adjacency[start])
    path = [start, first]
    while len(path) < len(atoms):
        previous, current = path[-2:]
        following = [neighbor for neighbor in adjacency[current] if neighbor != previous]
        if len(following) != 1 or following[0] in path:
            return None
        path.append(following[0])
    if start not in adjacency[path[-1]]:
        return None
    return tuple(path)


def _matching_locant_maps(
    mol: Molecule,
    cycle: tuple[int, ...],
    template: RetainedMonocycleGraphTemplate,
) -> tuple[dict[int, str], ...]:
    maps = []
    seen = set()
    for direction in (cycle, tuple(reversed(cycle))):
        for offset in range(len(direction)):
            oriented = direction[offset:] + direction[:offset]
            atom_to_locant = dict(zip(oriented, template.locants, strict=True))
            if not _atoms_match(mol, atom_to_locant, template):
                continue
            key = tuple(sorted(atom_to_locant.items()))
            if key not in seen:
                seen.add(key)
                maps.append(atom_to_locant)
    return tuple(maps)


def _atoms_match(
    mol: Molecule,
    atom_to_locant: dict[int, str],
    template: RetainedMonocycleGraphTemplate,
) -> bool:
    atom_by_locant = template.atom_by_locant
    atoms_match = all(
        mol.atoms[atom].symbol == atom_by_locant[locant].symbol
        and mol.atoms[atom].charge == atom_by_locant[locant].charge
        for atom, locant in atom_to_locant.items()
    )
    if not atoms_match:
        return False
    atom_by_locant_map = {locant: atom for atom, locant in atom_to_locant.items()}
    parent_atoms = set(atom_to_locant)
    for atom_id, locant in atom_to_locant.items():
        template_atom = atom_by_locant[locant]
        if template_atom.symbol == "C":
            continue
        if any(
            neighbor not in parent_atoms
            and (bond := mol.get_bond(atom_id, neighbor)) is not None
            and bond.order > 1
            for neighbor in mol.get_neighbors(atom_id)
        ):
            return False
    expected_double_edges = {frozenset(edge) for edge in template.double_bond_locants}
    for index, left_locant in enumerate(template.locants):
        right_locant = template.locants[(index + 1) % len(template.locants)]
        bond = mol.get_bond(atom_by_locant_map[left_locant], atom_by_locant_map[right_locant])
        expected_order = 2 if frozenset((left_locant, right_locant)) in expected_double_edges else 1
        if bond is None or bond.order != expected_order:
            return False
    return True


def _double_bond_count(mol: Molecule, atoms: set[int]) -> int:
    return sum(
        bond.order == 2 and bond.u in atoms and bond.v in atoms
        for bond in mol.bonds.values()
    )


def _has_cumulated_double_bond(mol: Molecule, atoms: set[int]) -> bool:
    return any(
        sum(
            (bond := mol.get_bond(atom, neighbor)) is not None and bond.order == 2
            for neighbor in mol.get_neighbors(atom)
            if neighbor in atoms
        )
        > 1
        for atom in atoms
    )
