"""Generic recognition of atomic and retained whole-graph components."""

from __future__ import annotations

from functools import lru_cache

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


def is_generic_atomic_charge_component(mol: Molecule, component_atoms: set[int]) -> bool:
    """Return whether a charged atom is named only by fallback charge notation."""

    if len(component_atoms) != 1:
        return False
    atom = mol.atoms[next(iter(component_atoms))]
    if atom.charge == 0 or atom.isotope is not None or atom.total_h_count != 0:
        return False
    if atom.charge > 0 and atom.symbol in RULES.ions.single_atom_cations:
        return False
    if atom.charge < 0 and atom.symbol in RULES.ions.single_atom_anions:
        return False
    return (atom.symbol, atom.charge, atom.total_h_count) not in RULES.ions.mononuclear_hydride_ions


def unsupported_generic_atomic_salt_components(mol: Molecule, components: list[set[int]]) -> tuple[set[int], ...]:
    """Return fallback atomic ions paired with an opposite-charge component.

    Atomic charge notation is unambiguous for an isolated ion or beside a
    structured ion. With an atomic counterion, OPSIN treats an unregistered
    element-plus-counterion phrase as a covalent composition instead, so the
    current grammar cannot safely represent it.
    """

    charges = [sum(mol.atoms[idx].charge for idx in component) for component in components]
    return tuple(
        component
        for component, charge in zip(components, charges, strict=True)
        if is_generic_atomic_charge_component(mol, component)
        and any(
            charge * other_charge < 0 and len(other_component) == 1
            for other_component, other_charge in zip(components, charges, strict=True)
        )
    )


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


@lru_cache(maxsize=1)
def _retained_component_templates() -> tuple[tuple[str, Molecule], ...]:
    """Parse the data-owned retained graphs once, outside registry construction."""

    from .graph_io import read_smiles

    return tuple((name, read_smiles(smiles)) for name, smiles in RULES.components.retained_component_graph_names)


def _component_graph_isomorphic(
    mol: Molecule,
    component_atoms: set[int],
    template: Molecule,
) -> bool:
    """Return whether two complete components are exactly label-isomorphic."""

    template_atoms = set(template.atoms)
    if len(component_atoms) != len(template_atoms):
        return False
    if component_graph_signature(mol, component_atoms) != component_graph_signature(template, template_atoms):
        return False

    def atom_label(graph: Molecule, atom_id: int) -> tuple[str, int, int | None]:
        atom = graph.atoms[atom_id]
        return atom.symbol, atom.charge, atom.isotope

    candidates = {
        atom_id: tuple(
            template_id
            for template_id in template_atoms
            if atom_label(mol, atom_id) == atom_label(template, template_id)
            and mol.degree(atom_id) == template.degree(template_id)
        )
        for atom_id in component_atoms
    }
    if any(not matches for matches in candidates.values()):
        return False
    ordered = tuple(sorted(component_atoms, key=lambda atom_id: (len(candidates[atom_id]), -mol.degree(atom_id))))

    def compatible(atom_id: int, template_id: int, mapping: dict[int, int]) -> bool:
        for mapped_atom, mapped_template in mapping.items():
            bond = mol.get_bond(atom_id, mapped_atom)
            template_bond = template.get_bond(template_id, mapped_template)
            if (bond is None) != (template_bond is None):
                return False
            if bond is not None and bond.order != template_bond.order:
                return False
        return True

    def extend(position: int, mapping: dict[int, int], used: set[int]) -> bool:
        if position == len(ordered):
            return True
        atom_id = ordered[position]
        for template_id in candidates[atom_id]:
            if template_id in used or not compatible(atom_id, template_id, mapping):
                continue
            mapping[atom_id] = template_id
            used.add(template_id)
            if extend(position + 1, mapping, used):
                return True
            used.remove(template_id)
            del mapping[atom_id]
        return False

    return extend(0, {}, set())


def retained_component_graph_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Return a retained name when the complete charged graph has a registered name."""

    if len(component_atoms) < 2:
        return ""
    # The registry describes parent graphs with implicit hydrogens. Carbon
    # hydrogens distinguish substituted organic molecules from these entries.
    if any(mol.atoms[idx].total_h_count for idx in component_atoms if mol.atoms[idx].symbol == "C"):
        return ""
    return next(
        (
            name
            for name, template in _retained_component_templates()
            if _component_graph_isomorphic(mol, component_atoms, template)
        ),
        "",
    )
