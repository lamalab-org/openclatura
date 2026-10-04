"""Generic recognition of atomic and retained whole-graph components."""

from __future__ import annotations

from .molecule import Molecule
from .nomenclature import RULES


def single_atom_component_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Return a rule-backed name for an isolated atom or mononuclear hydride."""

    if len(component_atoms) != 1:
        return ""
    atom = mol.atoms[next(iter(component_atoms))]
    if atom.symbol in RULES.ions.single_atom_cations and atom.charge > 0:
        return atom.element.name
    if atom.symbol in RULES.ions.single_atom_anions and atom.charge < 0:
        return RULES.ions.single_atom_anions[atom.symbol]

    ion_name = RULES.ions.mononuclear_hydride_ions.get((atom.symbol, atom.charge, atom.total_h_count))
    if ion_name:
        return ion_name

    if atom.charge == 0 and atom.total_h_count == atom.element.standard_valence:
        retained_name = RULES.components.retained_mononuclear_hydride_names.get(atom.symbol)
        if retained_name:
            return retained_name
        hydride_name = RULES.components.mononuclear_parent_hydrides.get(atom.symbol)
        if hydride_name:
            return hydride_name

    # P-72.2 charge notation preserves unusual isolated atoms rather than
    # silently coercing them to a common ion or parent hydride.
    if atom.isotope is None and atom.total_h_count == 0:
        if atom.charge == 0:
            return atom.element.name if atom.symbol == "H" else f"{atom.element.name}(0)"
        sign = "+" if atom.charge > 0 else "-"
        return f"{atom.element.name}({abs(atom.charge)}{sign})"
    return ""


def component_graph_signature(
    mol: Molecule, component_atoms: set[int]
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, str, int], ...]]:
    """Return the element/charge/bond multiset signature of a component."""

    atoms = tuple(sorted((mol.atoms[idx].symbol, int(mol.atoms[idx].charge)) for idx in component_atoms))
    bonds = []
    for bond in mol.bonds.values():
        if bond.u in component_atoms and bond.v in component_atoms:
            left, right = sorted((mol.atoms[bond.u].symbol, mol.atoms[bond.v].symbol))
            bonds.append((left, right, int(bond.order)))
    return atoms, tuple(sorted(bonds))


def retained_component_graph_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Return a retained name when the complete charged graph has a registered name."""

    if len(component_atoms) < 2:
        return ""
    # The registry describes parent graphs with implicit hydrogens. Carbon
    # hydrogens distinguish substituted organic molecules from these entries.
    if any(mol.atoms[idx].total_h_count for idx in component_atoms if mol.atoms[idx].symbol == "C"):
        return ""
    return RULES.components.retained_component_graph_names.get(component_graph_signature(mol, component_atoms), "")
