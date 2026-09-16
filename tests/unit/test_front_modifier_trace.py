"""Front modifiers retain recursive ownership without slowing plain naming."""

import pytest

from openclatura.assembly_parts import AssemblyParts
from openclatura.component_modifiers import add_component_front_modifiers
from openclatura.molecule import Molecule
from openclatura.namer import name_subgraph
from openclatura.perception import perceive_groups
from openclatura.trace_helpers import assembly_substituent_tree, assembly_trace_segments


@pytest.mark.parametrize("center", ["C", "S"])
@pytest.mark.parametrize("length", [1, 2, 4])
@pytest.mark.parametrize("include_trace", [False, True])
def test_front_modifier_trace_uses_one_scoped_recursive_call(center, length, include_trace):
    mol = Molecule()
    for index, symbol in enumerate(("C", center, "O", "O")):
        mol.add_atom(symbol, idx=index)
    mol.add_bond(0, 1)
    mol.add_bond(1, 2, order=2)
    mol.add_bond(1, 3)
    branch_atoms = set(range(4, 4 + length))
    previous = 3
    for atom in sorted(branch_atoms):
        mol.add_atom("C", idx=atom)
        mol.add_bond(previous, atom)
        previous = atom
    if center == "S":
        mol.add_atom("O", idx=4 + length)
        mol.add_bond(1, 4 + length, order=2)
    groups = perceive_groups(mol)
    group = next(g for g in groups if 3 in g.atoms_involved)
    excluded = set(mol.atoms) - branch_atoms
    calls = []

    def branch_namer(*args, **kwargs):
        calls.append(kwargs)
        return name_subgraph(*args, **kwargs)

    parts = AssemblyParts(parent_length=2)
    add_component_front_modifiers(
        mol,
        parts,
        groups,
        group.key,
        excluded,
        branch_namer,
        include_trace=include_trace,
    )
    assert len(calls) == 1
    assert parts.front_modifiers
    assert parts.front_modifier_atom_ids == branch_atoms
    assert bool(parts.front_modifier_items) == include_trace
    if not include_trace:
        assert calls == [{"upstream_atom": 3}]
        return
    item = parts.front_modifier_items[0]
    assert item.name == parts.front_modifiers[0]
    assert item.atom_ids == branch_atoms
    assert set(item.substituent_tree["atoms"]) == branch_atoms
    assert item.nested_decisions
    assert any(segment.get("nested_decisions") for segment in assembly_trace_segments(parts))
    tree = assembly_substituent_tree(parts, name=item.name, atom_ids=set(mol.atoms), mol=mol)
    assert len(tree["front_modifiers"]) == 1
