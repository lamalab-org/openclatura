"""Human-readable molecular structure descriptions from SMILES.

The describer reuses the naming pipeline's graph model, parent selection,
and numbering, then emits prose intended to be reconstructable by a
human. It is intentionally descriptive (graph facts in chemistry
vocabulary) rather than a second IUPAC name generator: a reader of the
description could draw the molecule.

Deterministic by construction. No LLM in the loop.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict, deque
from dataclasses import dataclass

from .chains import find_all_carbon_paths, find_ring_systems
from .engine import DEFAULT_NAMING_ENGINE, NamingRequest
from .graph_io import get_connected_components, read_smiles
from .locants import parse_locant
from .molecule import Molecule
from .namer import name_subgraph
from .numbering import number_parent
from .parent_selection import select_principal_parent

BOND_WORDS = {1: "single", 2: "double", 3: "triple"}
ELEMENT_WORDS = {
    "B": "boron",
    "C": "carbon",
    "N": "nitrogen",
    "O": "oxygen",
    "F": "fluorine",
    "Si": "silicon",
    "P": "phosphorus",
    "S": "sulfur",
    "Cl": "chlorine",
    "Se": "selenium",
    "Br": "bromine",
    "I": "iodine",
}
HALO_PREFIXES = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}


@dataclass(frozen=True)
class DescriptionFacts:
    """Structured facts extracted from a molecular component.

    Each field is a tuple of human-readable phrases the assembler
    composes into prose.
    """

    parent_summary: str
    parent_detail: str = ""
    heteroatoms: tuple[str, ...] = ()
    unsaturation: tuple[str, ...] = ()
    stereochemistry: tuple[str, ...] = ()
    substituents: tuple[str, ...] = ()
    connectivity: tuple[str, ...] = ()


@dataclass(frozen=True)
class Description:
    """Result of :func:`describe`.

    ``str(d)`` returns the prose. ``d.facts`` holds the structured
    per-component facts the prose was assembled from.
    """

    smiles: str
    name: str
    text: str
    facts: tuple[DescriptionFacts, ...] = ()

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "name": self.name,
            "text": self.text,
            "facts": [dataclasses.asdict(f) for f in self.facts],
        }


# --- public entry point --------------------------------------------------


def describe(smiles: str) -> Description:
    """Render a structure-driven natural-language description of ``smiles``."""

    mol = read_smiles(smiles)
    if not mol.atoms:
        return Description(smiles=smiles, name="", text="")

    components = get_connected_components(mol)
    name_result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles=smiles))

    facts_list: list[DescriptionFacts] = []
    sentences: list[str] = []
    for component in components:
        facts = _describe_component_facts(mol, component)
        if facts is None:
            sentences.append(_describe_unselected_component(mol, component))
            continue
        facts_list.append(facts)
        sentences.append(_assemble_description(facts))

    return Description(
        smiles=smiles,
        name=name_result.name,
        text=" ".join(part for part in sentences if part),
        facts=tuple(facts_list),
    )


# --- per-component fact extraction ---------------------------------------


def _describe_component_facts(mol: Molecule, component_atoms: set[int]) -> DescriptionFacts | None:
    """Build the structured facts for one connected component."""

    if len(component_atoms) == 1:
        atom = mol.atoms[next(iter(component_atoms))]
        charge_text = " cation" if atom.charge > 0 else " anion" if atom.charge < 0 else ""
        return DescriptionFacts(parent_summary=f"a single {atom.element.name}{charge_text}")

    exclude_atoms = set(mol.atoms.keys()) - component_atoms
    parent = _select_description_parent(mol, exclude_atoms)
    if parent is None:
        return None

    seed_path = parent.primary_path
    branch_index = _branch_attachment_index(mol, seed_path, exclude_atoms)
    numbered_path = number_parent(
        mol,
        parent.paths,
        set(),
        branch_index,
        parent.is_ring,
        parent.is_bicycle,
        parent.is_spiro,
        is_polycycle=parent.is_polycycle,
        fixed_start=parent.is_bicycle or parent.is_spiro or parent.is_polycycle,
    )
    locants = {atom_idx: str(i + 1) for i, atom_idx in enumerate(numbered_path)}
    parent_set = set(numbered_path)

    return DescriptionFacts(
        parent_summary=_parent_skeleton_phrase(
            mol, numbered_path, parent.is_ring, parent.is_bicycle, parent.is_spiro, parent.is_polycycle
        ),
        parent_detail=_parent_saturation_phrase(mol, numbered_path, parent.is_ring),
        heteroatoms=tuple(_parent_heteroatom_phrases(mol, numbered_path, locants)),
        unsaturation=tuple(_parent_unsaturation_phrases(mol, numbered_path, locants)),
        stereochemistry=tuple(_stereochemistry_phrases(mol, numbered_path, locants)),
        substituents=tuple(_parent_substituent_phrases(mol, numbered_path, parent_set, exclude_atoms, locants)),
        connectivity=tuple(_parent_connection_phrases(mol, numbered_path, locants, parent.is_ring)),
    )


def _describe_unselected_component(mol: Molecule, component: set[int]) -> str:
    """Fall-back prose when no parent could be selected."""

    n = len(component)
    elements = sorted({mol.atoms[i].symbol for i in component})
    return f"A {n}-atom fragment with elements {{{', '.join(elements)}}} (no parent skeleton selected)."


def _select_description_parent(mol: Molecule, exclude_atoms: set[int]):
    chains = find_all_carbon_paths(mol, exclude_atoms)
    ring_systems = find_ring_systems(mol, exclude_atoms)
    if not chains and not ring_systems:
        return None
    return select_principal_parent(mol, chains, ring_systems, [])


def _branch_attachment_index(mol: Molecule, parent_path: list[int], exclude_atoms: set[int]) -> dict[int, list[str]]:
    """Approximate branch labels so numbering ranks them consistently."""

    parent_set = set(parent_path)
    mapping: dict[int, list[str]] = defaultdict(list)
    for atom_idx in parent_path:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in parent_set or neighbor in exclude_atoms:
                continue
            mapping[atom_idx].append(_short_substituent_label(mol, atom_idx, neighbor, parent_set))
    return dict(mapping)


# --- skeletal-fact phrases -----------------------------------------------


def _parent_skeleton_phrase(
    mol: Molecule,
    numbered_path: list[int],
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
) -> str:
    size = len(numbered_path)
    hetero = any(not mol.atoms[idx].is_carbon for idx in numbered_path)
    if is_bicycle:
        kind = "bicyclic heteroskeleton" if hetero else "bicyclic carbon skeleton"
    elif is_spiro:
        kind = "spiro heteroskeleton" if hetero else "spiro carbon skeleton"
    elif is_polycycle:
        kind = "polycyclic heteroskeleton" if hetero else "polycyclic carbon skeleton"
    elif is_ring:
        kind = "heterocycle" if hetero else "carbocycle"
    else:
        kind = "heteroatom-containing chain" if hetero else "carbon chain"
    measure = "membered" if (is_ring or is_bicycle or is_spiro or is_polycycle) else "atom"
    return f"a {size}-{measure} {kind}"


def _parent_heteroatom_phrases(mol: Molecule, numbered_path: list[int], locants: dict[int, str]) -> list[str]:
    by_element: dict[str, list[str]] = defaultdict(list)
    for atom_idx in numbered_path:
        atom = mol.atoms[atom_idx]
        if not atom.is_carbon:
            by_element[ELEMENT_WORDS.get(atom.symbol, atom.symbol)].append(locants[atom_idx])
    return [f"{element} at {_positions_text(locs)}" for element, locs in sorted(by_element.items())]


def _parent_unsaturation_phrases(mol: Molecule, numbered_path: list[int], locants: dict[int, str]) -> list[str]:
    phrases: list[str] = []
    seen: set[int] = set()
    parent_set = set(numbered_path)
    for atom_idx in numbered_path:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in parent_set:
                continue
            bond = mol.get_bond(atom_idx, neighbor)
            if not bond or bond.idx in seen or bond.order <= 1:
                continue
            seen.add(bond.idx)
            loc_pair = sorted([locants[atom_idx], locants[neighbor]], key=parse_locant)
            stereo = f" with {bond.stereo} geometry" if bond.stereo else ""
            phrases.append(
                f"a {BOND_WORDS.get(bond.order, bond.order)} bond between positions "
                f"{loc_pair[0]} and {loc_pair[1]}{stereo}"
            )
    return phrases


def _parent_saturation_phrase(mol: Molecule, numbered_path: list[int], is_ring: bool) -> str:
    locants = {idx: str(i + 1) for i, idx in enumerate(numbered_path)}
    if _parent_unsaturation_phrases(mol, numbered_path, locants):
        return ""
    return "all parent ring bonds are single" if is_ring else "all parent-chain bonds are single"


def _stereochemistry_phrases(mol: Molecule, numbered_path: list[int], locants: dict[int, str]) -> list[str]:
    return [f"position {locants[idx]} is {mol.atoms[idx].stereo}" for idx in numbered_path if mol.atoms[idx].stereo]


def _parent_substituent_phrases(
    mol: Molecule,
    numbered_path: list[int],
    parent_set: set[int],
    exclude_atoms: set[int],
    locants: dict[int, str],
) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    side_chains: list[tuple[str, str]] = []
    handled: set[int] = set()

    for atom_idx in numbered_path:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in parent_set or neighbor in exclude_atoms or neighbor in handled:
                continue
            label = _simple_substituent_description(mol, atom_idx, neighbor, parent_set)
            branch_atoms = _branch_component(mol, neighbor, parent_set | exclude_atoms)
            handled.update(branch_atoms)
            if label:
                grouped[label].append(locants[atom_idx])
            else:
                side_chains.append((locants[atom_idx], _describe_side_chain(mol, neighbor, parent_set | exclude_atoms)))

    parts: list[str] = []
    for label, positions in sorted(grouped.items(), key=lambda item: parse_locant(item[1][0])):
        parts.append(f"{_pluralize_substituent(label, len(positions))} at {_positions_text(positions)}")
    for locant, body in sorted(side_chains, key=lambda item: parse_locant(item[0])):
        parts.append(f"a side chain at position {locant}; this side chain has {body}")
    return parts


def _parent_connection_phrases(
    mol: Molecule, numbered_path: list[int], locants: dict[int, str], is_ring: bool
) -> list[str]:
    if not numbered_path:
        return []
    connections: list[str] = []
    for i in range(len(numbered_path) - 1):
        u, v = numbered_path[i], numbered_path[i + 1]
        bond = mol.get_bond(u, v)
        if bond:
            connections.append(f"{locants[u]}-{locants[v]} {BOND_WORDS.get(bond.order, bond.order)}")
    if is_ring and len(numbered_path) > 2:
        u, v = numbered_path[-1], numbered_path[0]
        bond = mol.get_bond(u, v)
        if bond:
            connections.append(f"{locants[u]}-{locants[v]} {BOND_WORDS.get(bond.order, bond.order)}")
    return connections


# --- prose assembler -----------------------------------------------------


def _assemble_description(facts: DescriptionFacts) -> str:
    sentences: list[str] = []
    opening = f"The molecule is built around {facts.parent_summary}"
    if facts.heteroatoms:
        opening += f", with {_join_human(list(facts.heteroatoms))}"
    sentences.append(opening + ".")

    if facts.unsaturation:
        sentences.append(f"Within that parent framework, there is {_join_human(list(facts.unsaturation))}.")
    elif facts.parent_detail:
        sentences.append(f"Within that parent framework, {facts.parent_detail}.")

    if facts.substituents:
        verb = "is" if len(facts.substituents) == 1 else "are"
        sentences.append(f"Attached to the parent {verb} {_join_human(list(facts.substituents))}.")
    else:
        sentences.append("There are no substituents outside the parent framework.")

    if facts.stereochemistry:
        sentences.append(f"The specified stereochemistry is: {'; '.join(facts.stereochemistry)}.")

    if facts.connectivity:
        sentences.append(f"For reconstruction, number the parent as follows: {', '.join(facts.connectivity)}.")

    return " ".join(sentences)


# --- substituent / branch helpers ----------------------------------------


def _short_substituent_label(mol: Molecule, parent_idx: int, neighbor_idx: int, parent_set: set[int]) -> str:
    label = _simple_substituent_description(mol, parent_idx, neighbor_idx, parent_set)
    if label:
        return label
    try:
        return name_subgraph(mol, neighbor_idx, parent_set, upstream_atom=parent_idx)
    except Exception:
        return mol.atoms[neighbor_idx].symbol


def _simple_substituent_description(mol: Molecule, parent_idx: int, neighbor_idx: int, parent_set: set[int]) -> str:
    atom = mol.atoms[neighbor_idx]
    bond = mol.get_bond(parent_idx, neighbor_idx)
    if atom.symbol == "O":
        externals = [n for n in mol.get_neighbors(neighbor_idx) if n != parent_idx and n not in parent_set]
        if bond and bond.order == 2:
            return "oxo group"
        if not externals and atom.charge == -1:
            return "oxido group"
        if not externals:
            return "hydroxy group"
    if atom.symbol == "N":
        externals = [n for n in mol.get_neighbors(neighbor_idx) if n != parent_idx and n not in parent_set]
        if bond and bond.order == 2:
            return "imino group"
        if bond and bond.order == 3:
            return "nitrilo group"
        if not externals:
            return "amino group"
    if atom.symbol == "S":
        externals = [n for n in mol.get_neighbors(neighbor_idx) if n != parent_idx and n not in parent_set]
        if bond and bond.order == 2:
            return "thioxo group"
        if not externals:
            return "sulfanyl group"
    if atom.symbol in HALO_PREFIXES and mol.degree(neighbor_idx) == 1:
        return f"{HALO_PREFIXES[atom.symbol]} group"
    return ""


def _branch_component(mol: Molecule, start_idx: int, blocked_atoms: set[int]) -> set[int]:
    component: set[int] = set()
    queue: deque[int] = deque([start_idx])
    while queue:
        current = queue.popleft()
        if current in component or current in blocked_atoms:
            continue
        component.add(current)
        for neighbor in mol.get_neighbors(current):
            if neighbor not in component and neighbor not in blocked_atoms:
                queue.append(neighbor)
    return component


def _describe_side_chain(mol: Molecule, start_idx: int, blocked_atoms: set[int]) -> str:
    component = _branch_component(mol, start_idx, blocked_atoms)
    if not component:
        return "empty branch"

    simple = _simple_side_chain_summary(mol, start_idx, component, blocked_atoms)
    if simple:
        return simple

    local_order = _local_branch_order(mol, start_idx, component)
    local_locants = {atom_idx: str(i + 1) for i, atom_idx in enumerate(local_order)}
    carbon_count = sum(1 for idx in component if mol.atoms[idx].is_carbon)
    hetero_count = len(component) - carbon_count

    atoms = [
        f"branch atom {local_locants[idx]} is {ELEMENT_WORDS.get(mol.atoms[idx].symbol, mol.atoms[idx].symbol)}"
        for idx in local_order
    ]
    bonds: list[str] = []
    seen: set[int] = set()
    for atom_idx in local_order:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in component:
                continue
            bond = mol.get_bond(atom_idx, neighbor)
            if bond and bond.idx not in seen:
                seen.add(bond.idx)
                bonds.append(
                    f"{local_locants[atom_idx]}-{local_locants[neighbor]} {BOND_WORDS.get(bond.order, bond.order)}"
                )

    count_text = f"{carbon_count} carbon atom{'s' if carbon_count != 1 else ''}"
    if hetero_count:
        count_text += f" and {hetero_count} heteroatom{'s' if hetero_count != 1 else ''}"
    return f"{count_text}; " + "; ".join(atoms) + "; branch connectivity: " + ", ".join(bonds)


def _local_branch_order(mol: Molecule, start_idx: int, component: set[int]) -> list[int]:
    order: list[int] = []
    queue: deque[int] = deque([start_idx])
    while queue:
        current = queue.popleft()
        if current in order or current not in component:
            continue
        order.append(current)
        for neighbor in sorted(mol.get_neighbors(current)):
            if neighbor in component and neighbor not in order:
                queue.append(neighbor)
    return order


def _simple_side_chain_summary(mol: Molecule, start_idx: int, component: set[int], blocked_atoms: set[int]) -> str:
    """Compact prose for common one-carbon side chains."""

    carbon_atoms = [idx for idx in component if mol.atoms[idx].is_carbon]
    if len(carbon_atoms) != 1 or carbon_atoms[0] != start_idx:
        return ""
    attached = [n for n in mol.get_neighbors(start_idx) if n in component and n != start_idx]
    if not attached:
        return "1 carbon atom"
    hydroxy = [
        n
        for n in attached
        if mol.atoms[n].symbol == "O"
        and (bond := mol.get_bond(start_idx, n))
        and bond.order == 1
        and all(x == start_idx or x in blocked_atoms for x in mol.get_neighbors(n))
    ]
    if hydroxy and len(hydroxy) == len(attached):
        return "1 carbon atom connected to a hydroxy group"
    return ""


# --- formatting ----------------------------------------------------------


def _positions_text(locants: list[str]) -> str:
    ordered = sorted([str(loc) for loc in locants], key=parse_locant)
    if len(ordered) == 1:
        return f"position {ordered[0]}"
    return "positions " + _join_human(ordered)


def _join_human(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _pluralize_substituent(label: str, count: int) -> str:
    if count == 1:
        return _indefinite(label)
    if label.endswith(" group"):
        return label[:-6] + " groups"
    return label + "s"


def _indefinite(text: str) -> str:
    if text.startswith(("a ", "an ", "the ")):
        return text
    article = "an" if text[0].lower() in "aeiou" else "a"
    return f"{article} {text}"


__all__ = [
    "Description",
    "DescriptionFacts",
    "describe",
]
