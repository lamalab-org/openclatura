"""Substitution preserves graph-proved added-H parent nitrogen ownership."""

from dataclasses import replace
from random import Random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import _audit_derivative_state, _has_consistent_derivative_operations
from openclatura.fusion.mancude import is_added_hydrogen_nitrogen, parent_derivative_state
from openclatura.fusion.model import FusionConfirmed, FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol


def _graph(family, donor_length, external):
    graph = Chem.RWMol()
    symbols = ["N", *(["C"] * (11 if family == "azocine" else 13))]
    if family == "diazepine":
        symbols[6] = "N"
    for symbol in symbols:
        graph.AddAtom(Chem.Atom(symbol))
    if family == "azocine":
        edges = (
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 6, 1),
            (6, 7, 2),
            (7, 8, 1),
            (8, 9, 2),
            (9, 10, 1),
            (10, 11, 2),
            (11, 6, 1),
            (11, 0, 1),
        )
        external_sites = (5,)
    else:
        edges = (
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 6, 1),
            (6, 2, 1),
            (6, 7, 1),
            (7, 8, 1),
            (8, 9, 2),
            (9, 10, 1),
            (10, 11, 2),
            (11, 12, 1),
            (12, 13, 2),
            (13, 8, 1),
            (13, 0, 1),
        )
        external_sites = (1, 7)
    for u, v, order in edges:
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    for parent in external_sites:
        graph.AddBond(parent, graph.AddAtom(Chem.Atom(external)), Chem.BondType.DOUBLE)
    last = 0
    for _ in range(donor_length):
        other = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, other, Chem.BondType.SINGLE)
        last = other
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol, frozenset(range(len(symbols)))


def _permutations(graph):
    order = list(range(graph.GetNumAtoms()))
    shuffled = order[:]
    Random(54374).shuffle(shuffled)
    return order, order[::-1], shuffled


def _state(graph, atoms, locants):
    mol = read_rdkit_mol(graph)
    parent = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, mol.atoms[atom].symbol) for atom in sorted(atoms)),
        bonds=tuple(
            FusionGraphBond(tuple(sorted((bond.u, bond.v))), "mancude")
            for bond in mol.bonds.values()
            if bond.u in atoms and bond.v in atoms
        ),
    )
    model = parent_bond_model(parent)
    return mol, parent_derivative_state(mol, atoms, model, locants)


@pytest.mark.parametrize("family", ("azocine", "diazepine"))
@pytest.mark.parametrize("donor_length", (0, 1, 2))
@pytest.mark.parametrize("external", ("O", "N", "C"))
def test_substituted_nitrogen_parent_h_survives_external_pi_composition(family, donor_length, external):
    graph, core = _graph(family, donor_length, external)
    for order in _permutations(graph):
        atoms = frozenset(atom for atom, original in enumerate(order) if original in core)
        locants = {atom: str(order[atom] + 1) for atom in atoms}
        mol, state = _state(Chem.RenumberAtoms(graph, order), atoms, locants)
        assert state is not None
        assert not state.unsaturation_operations
        added = {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids}
        assert {order[atom] for atom in added} == {0}
        assert all(is_added_hydrogen_nitrogen(mol, atoms, atom) for atom in added)
        assert all(mol.atoms[atom].total_h_count == (donor_length == 0) for atom in added)
        assert all(order == 1 for edge, order in state.bond_delta.assignment.orders if added.intersection(edge))
        hydro = {atom for operation in state.hydro_operations for atom in operation.atom_ids}
        assert {order[atom] for atom in hydro} == ({2, 5} if family == "diazepine" else set())
        assert not added.intersection(hydro)
        assert not added.intersection(operation.parent_atom_id for operation in state.external_pi_operations)
        if family == "diazepine":
            assert state.pi_redistribution is not None
            assert state.pi_redistribution.hydrogenated_atom_ids == hydro
            assert state.bond_delta.composition_model is not None


@pytest.mark.parametrize("corruption", ("charge", "aromatic", "extra_h", "external_pi", "outside", "junction"))
def test_added_nitrogen_requires_exact_neutral_two_connected_sigma_valence(corruption):
    graph, atoms = _graph("azocine", 1, "O")
    mol = read_rdkit_mol(graph)
    assert is_added_hydrogen_nitrogen(mol, atoms, 0)
    if corruption == "charge":
        mol.update_atom(0, charge=1)
    elif corruption == "aromatic":
        mol.update_atom(0, is_aromatic=True)
    elif corruption == "extra_h":
        mol.update_atom(0, explicit_h_count=1, total_h_count=1)
    elif corruption == "external_pi":
        external = next(atom for atom in mol.get_neighbors(0) if atom not in atoms)
        mol.update_bond(mol.get_bond(0, external).idx, order=2)
    elif corruption == "outside":
        atoms = atoms - {0}
    else:
        atoms = atoms | set(mol.get_neighbors(0))
    assert not is_added_hydrogen_nitrogen(mol, atoms, 0)


def _audited_plan(graph):
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    assert result.plan.audit.confirmed
    return mol, atoms, result.plan


def _assert_exact_fusion(graph, expected_added, expected_indicated=frozenset()):
    mol, atoms, plan = _audited_plan(graph)
    state = plan.derivative_state
    added = {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids}
    assert added == expected_added
    assert all(is_added_hydrogen_nitrogen(mol, atoms, atom) for atom in added if mol.atoms[atom].symbol == "N")
    locants = dict(plan.numbering.input_locant_maps[0])
    assert {atom for atom, locant in locants.items() if locant in plan.indicated_hydrogens} == expected_indicated
    name = name_mol(graph, include_trace=True)
    assert name.error is None
    assert name.parent_nomenclature == "systematic_fusion"
    check = verify_with_opsin(name.name, Chem.MolToSmiles(graph), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip == Chem.MolToSmiles(graph)
    return name.name


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="OPSIN requires Java and py2opsin")
@pytest.mark.parametrize("family", ("azocine", "diazepine"))
@pytest.mark.parametrize("donor_length", (0, 1, 2))
@pytest.mark.parametrize("external", ("O", "N", "C"))
def test_graph_built_substituted_parent_h_exact_fusion_roundtrip(family, donor_length, external):
    graph, _ = _graph(family, donor_length, external)
    # The component's existing intrinsic N-H convention precedes added H.
    intrinsic_nh = family == "diazepine" and donor_length == 0
    names = {
        _assert_exact_fusion(
            Chem.RenumberAtoms(graph, order),
            {order.index(atom) for atom in ({2, 5} if intrinsic_nh else {0})},
            {order.index(0)} if intrinsic_nh else set(),
        )
        for order in _permutations(graph)
    }
    assert len(names) == 1


@pytest.mark.parametrize(
    "corruption",
    (
        "charge",
        "aromatic",
        "extra_h",
        "external_pi",
        "internal_pi",
        "duplicate_hydro",
        "duplicate_indicated",
        "indicated",
    ),
)
def test_added_nitrogen_audit_preserves_valence_ownership_and_indicated_h_rules(corruption):
    graph, _ = _graph("azocine", 1, "O")
    mol, atoms, plan = _audited_plan(graph)
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    state = plan.derivative_state
    indicated = plan.indicated_hydrogens
    assert _has_consistent_derivative_operations(mol, plan.ast, specs, plan.numbering, indicated, state)
    assert not indicated
    (operation,) = state.added_hydrogen_operations
    assert operation.atom_ids == (0,)
    if corruption == "charge":
        mol.update_atom(0, charge=1)
    elif corruption == "aromatic":
        mol.update_atom(0, is_aromatic=True)
    elif corruption == "extra_h":
        mol.update_atom(0, total_h_count=1, explicit_h_count=1)
    elif corruption == "external_pi":
        external = next(atom for atom in mol.get_neighbors(0) if atom not in atoms)
        mol.update_bond(mol.get_bond(0, external).idx, order=2)
    elif corruption == "internal_pi":
        assignment = replace(
            state.bond_delta.assignment,
            orders=tuple((edge, 2 if 0 in edge else order) for edge, order in state.bond_delta.assignment.orders),
        )
        state = replace(state, bond_delta=replace(state.bond_delta, assignment=assignment))
    elif corruption == "duplicate_hydro":
        state = replace(state, hydro_operations=(replace(operation, operation_kind="additive_hydrogen"),))
    else:
        indicated = (dict(plan.numbering.input_locant_maps[0])[0],)
        if corruption == "indicated":
            state = replace(state, bond_delta=replace(state.bond_delta, added_hydrogen_operations=()))
    assert not _has_consistent_derivative_operations(mol, plan.ast, specs, plan.numbering, indicated, state)


@pytest.mark.parametrize("corruption", ("locant", "atom", "bond", "missing"))
def test_added_nitrogen_independent_replay_requires_exact_operation(corruption):
    graph, _ = _graph("azocine", 1, "O")
    mol, atoms, plan = _audited_plan(graph)
    state = plan.derivative_state
    errors = []
    _audit_derivative_state(mol, atoms, plan.numbering, plan.bond_model, plan.indicated_hydrogens, state, errors)
    assert not errors
    (operation,) = state.added_hydrogen_operations
    if corruption == "locant":
        operation = replace(operation, locants=("99",))
    elif corruption == "atom":
        operation = replace(operation, atom_ids=(1,))
    elif corruption == "bond":
        operation = replace(operation, bond_ids=())
    state = replace(
        state,
        bond_delta=replace(state.bond_delta, added_hydrogen_operations=() if corruption == "missing" else (operation,)),
    )
    _audit_derivative_state(mol, atoms, plan.numbering, plan.bond_model, plan.indicated_hydrogens, state, errors)
    assert "typed derivative state does not carry the selected parent bond delta" in errors


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="OPSIN requires Java and py2opsin")
@pytest.mark.parametrize(
    "smiles",
    (
        pytest.param("COc1cc2c(cc1CCO)N(C)C(=O)[C@H]1C=C(c3ccc(C)cc3)CN1C2=O", id="54374"),
        pytest.param("C=C1/C=C(c2cccc3c2O/C=C\\C=C/C3=C)\\C=C/N(C)c2ccccc21", id="64192"),
    ),
)
def test_pubchem_substituted_nitrogen_added_h_exact_fusion_roundtrip(smiles):
    graph = Chem.MolFromSmiles(smiles)
    donor = next(
        atom.GetIdx()
        for atom in graph.GetAtoms()
        if atom.GetSymbol() == "N" and sum(neighbor.IsInRing() for neighbor in atom.GetNeighbors()) == 2
    )
    names = {
        _assert_exact_fusion(Chem.RenumberAtoms(graph, order), {order.index(donor)}) for order in _permutations(graph)
    }
    assert len(names) == 1
