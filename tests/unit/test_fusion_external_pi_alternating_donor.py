"""External-pi composition retains exact alternating donor/hydro witnesses."""

from dataclasses import replace
from random import Random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.mancude import prove_pi_redistribution
from openclatura.fusion.model import BondAssignment, FusionConfirmed
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol

CASES = (
    pytest.param(
        "C/C=C/Cn1c(=O)c2c(nc3n2CC(C(C)(C)C)=NN3)n(C)c1=O",
        ("1", "4"),
        id="48596",
    ),
    pytest.param(
        "CC=CCc1nn(C)c2nc3n(c2c1=O)C(C)C(C)=NN3CC",
        ("6", "9"),
        id="52576",
    ),
    pytest.param(
        "C=C(C)Cn1c(=O)c2c(nc3n2C(C)C(C)=NN3)n(C)c1=O",
        ("1", "4"),
        id="75361",
    ),
)


def _graph(donor_length, carbon_methyl, external):
    graph = Chem.RWMol()
    for symbol in ("N", "C", "C", "C", "N", "C", "N", "C", "C", "N", "N", "N", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    for u, v, order in (
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 1),
        (4, 5, 2),
        (5, 6, 1),
        (6, 2, 1),
        (6, 7, 1),
        (7, 8, 1),
        (8, 9, 2),
        (9, 10, 1),
        (10, 5, 1),
        (3, 11, 1),
        (11, 12, 1),
        (12, 0, 1),
    ):
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    for parent in (1, 12):
        graph.AddBond(parent, graph.AddAtom(Chem.Atom(external)), Chem.BondType.DOUBLE)
    for parent in (0, 11):
        graph.AddBond(parent, graph.AddAtom(Chem.Atom("C")), Chem.BondType.SINGLE)
    last = 10
    for _ in range(donor_length):
        other = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, other, Chem.BondType.SINGLE)
        last = other
    if carbon_methyl:
        graph.AddBond(7, graph.AddAtom(Chem.Atom("C")), Chem.BondType.SINGLE)
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _permutations(mol):
    original = list(range(mol.GetNumAtoms()))
    shuffled = original[:]
    Random(315).shuffle(shuffled)
    return (original, original[::-1], shuffled)


def _plan(rd_mol):
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    assert result.plan.audit.confirmed
    return mol, atoms, result.plan


def _assert_witness(mol, atoms, plan):
    state = plan.derivative_state
    delta = state.bond_delta
    proof = state.pi_redistribution
    assert proof is not None
    assert delta.additional_multiple_bond_ids
    assert delta.composition_model is not None
    assert delta.assignment in delta.composition_model.allowed_kekule_assignments
    assert not state.unsaturation_operations
    assert not state.added_hydrogen_operations
    hydro = {atom for op in state.hydro_operations for atom in op.atom_ids}
    assert hydro == proof.hydrogenated_atom_ids
    assert len(hydro) == 2
    assert {mol.atoms[atom].symbol for atom in hydro} == {"C", "N"}
    assert all(not mol.atoms[atom].is_aromatic for atom in hydro)
    assert len(hydro) == 2 * (len(proof.removed_bond_ids) - len(proof.added_bond_ids))
    assert proof.final_assignment == BondAssignment(
        tuple((edge, mol.get_bond(*edge).order) for edge, _ in delta.assignment.orders)
    )
    assert (
        prove_pi_redistribution(
            mol,
            atoms,
            plan.bond_model,
            delta,
            oxo_operations=state.oxo_operations,
            imino_operations=state.imino_operations,
        )
        == proof
    )
    for operation in state.external_pi_operations:
        assert operation.parent_atom_id not in hydro
        assert operation.bond_id not in proof.removed_bond_ids | proof.added_bond_ids


@pytest.mark.parametrize("external", ("O", "N"))
@pytest.mark.parametrize("donor_length", (0, 1, 2))
@pytest.mark.parametrize("carbon_methyl", (False, True))
def test_graph_built_external_pi_donor_shift_is_permutation_invariant(external, donor_length, carbon_methyl):
    graph = _graph(donor_length, carbon_methyl, external)
    names = []
    for order in _permutations(graph):
        rd_mol = Chem.RenumberAtoms(graph, order)
        mol, atoms, plan = _plan(rd_mol)
        _assert_witness(mol, atoms, plan)
        assert {order[atom] for op in plan.derivative_state.hydro_operations for atom in op.atom_ids} == {7, 10}
        result = name_mol(rd_mol, include_trace=True)
        assert result.parent_nomenclature == "systematic_fusion"
        assert result.error is None
        names.append(result.name)
        if opsin_available():
            check = verify_with_opsin(result.name, Chem.MolToSmiles(graph), standardize_smiles=False)
            assert check.status == "matched", check.to_dict()
            assert check.canonical_original == check.canonical_roundtrip
    assert len(set(names)) == 1


@pytest.mark.opsin
@pytest.mark.parametrize("smiles,hydro", CASES)
def test_pubchem_external_pi_donor_shift_exact_roundtrip(smiles, hydro):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    graph = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(graph)
    names = []
    for order in _permutations(graph):
        rd_mol = Chem.RenumberAtoms(graph, order)
        mol, atoms, plan = _plan(rd_mol)
        _assert_witness(mol, atoms, plan)
        assert plan.derivative_state.hydro_operations[0].locants == hydro
        result = name_mol(rd_mol, include_trace=True)
        assert result.parent_nomenclature == "systematic_fusion"
        assert result.error is None
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip == canonical
        assert Chem.MolToSmiles(Chem.MolFromSmiles(check.opsin_smiles)) == canonical
        names.append(result.name)
    assert len(set(names)) == 1


@pytest.mark.parametrize("corruption", ("charged_donor", "external_bond", "required_double", "missing_model"))
def test_alternating_composition_still_requires_exact_valence_and_bond_domain(corruption):
    mol, atoms, plan = _plan(_graph(0, False, "O"))
    state = plan.derivative_state
    delta = state.bond_delta
    _assert_witness(mol, atoms, plan)
    # Intrinsic-H completion may already constrain plan.bond_model. Use the
    # original component domain to test loss of the external-pi domain witness.
    uncomposed_model = parent_bond_model(plan.abstract_parent_graph)
    assert (
        prove_pi_redistribution(mol, atoms, uncomposed_model, delta, oxo_operations=state.oxo_operations)
        == state.pi_redistribution
    )
    if corruption == "charged_donor":
        mol.update_atom(10, charge=1)
    elif corruption == "external_bond":
        mol.update_bond(state.oxo_operations[0].bond_id, order=1)
    elif corruption == "missing_model":
        assert delta.assignment not in uncomposed_model.allowed_kekule_assignments
        delta = replace(delta, composition_model=None)
    else:
        model = delta.composition_model
        edge = next(edge for edge, order in delta.assignment.orders if order == 2 and 10 in edge)
        delta = replace(
            delta,
            composition_model=replace(
                model,
                required_double_bonds=model.required_double_bonds | {edge},
                pi_eligible_edges=model.pi_eligible_edges - {edge},
            ),
        )
    assert (
        prove_pi_redistribution(
            mol,
            atoms,
            uncomposed_model,
            delta,
            oxo_operations=state.oxo_operations,
        )
        is None
    )
