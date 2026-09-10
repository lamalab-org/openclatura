"""Replacement fusion must prove the restored carbon/pi/H state jointly."""

import random
from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionMode
from openclatura.fusion.replacement_state import prove_replacement_state
from openclatura.fusion.third_component import plan_third_component_fusion_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.opsin_verify import verify_with_opsin

CASES = (
    pytest.param("C[C@H]1CC[C@@]2(C3(C)OCCO3)CC[C@H]3CON1[C@H]32", 1, id="pubchem-19119"),
    pytest.param(
        "COC(=O)[C@@H]1ON2O[C@@H](O[C@@H]3CCCC[C@H]3c3ccccc3)[C@H](C)[C@H]3CC[C@@H]1[C@]32C",
        0,
        id="pubchem-40352",
    ),
)


def _plan(graph):
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    plan = plan_third_component_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert plan is not None
    assert plan.parent.is_skeletal_replacement_fusion
    assert plan.parent.audit_ok
    assert plan.parent.replacement_fusion_state is not None
    return mol, plan.parent


def _orders(size):
    original = list(range(size))
    yield original
    yield list(reversed(original))
    for seed in (7, 73, 19119, 40352):
        shuffled = original.copy()
        random.Random(seed).shuffle(shuffled)
        yield shuffled


@pytest.mark.parametrize("smiles,indicated_count", CASES)
def test_replacement_proof_accounts_for_every_atom_without_changing_carbon_certificate(smiles, indicated_count):
    mol, parent = _plan(Chem.MolFromSmiles(smiles))
    state = parent.replacement_fusion_state
    carbon = parent.fusion_plan
    original = deepcopy(carbon)
    assert state.audit_ok
    assert parent.bond_model is state.bond_model
    assert parent.derivative_state is state.derivative_state
    assert carbon.bond_model.maximum_non_cumulative_double_bonds == 5
    assert state.bond_model.maximum_non_cumulative_double_bonds == 4
    assert len(state.indicated_hydrogens) == indicated_count
    assert parent.hydride_metadata.default_indicated_h == state.indicated_hydrogens
    assert "joint_carbon_pi_hydrogen_state" in parent.skeletal_replacement_audit_checks
    delta = state.derivative_state.bond_delta
    assert len(delta.hydrogenated_edges) == 4
    assert len(delta.hydrogenated_atom_ids) == 8
    assert all(mol.atoms[atom].is_carbon for atom in delta.hydrogenated_atom_ids)
    assert {row.atom_id for row in state.hydrogen_balances} == parent.atoms
    for row in state.hydrogen_balances:
        assert row.exact
        assert row.observed_hydrogens == mol.atoms[row.atom_id].total_h_count
        assert row.parent_hydrogens + row.additive_hydrogens == row.external_bond_order + row.observed_hydrogens
    assert {atom.id: atom.symbol for atom in state.graph.atoms} == {
        atom: mol.atoms[atom].symbol for atom in parent.atoms
    }
    assert all(atom.symbol == "C" for atom in carbon.abstract_parent_graph.atoms)
    assert prove_replacement_state(mol, carbon) == state
    assert carbon == original


@pytest.mark.parametrize("smiles,indicated_count", CASES)
def test_replacement_names_and_hydrogen_ownership_are_permutation_invariant(smiles, indicated_count):
    graph = Chem.MolFromSmiles(smiles)
    names = set()
    for order in _orders(graph.GetNumAtoms()):
        permuted = Chem.RenumberAtoms(graph, order)
        result = name_mol(permuted, include_trace=True, verify_self=True, token_debug=True)
        assert result.error is None
        assert result.parent_nomenclature == "skeletal_replacement_fusion"
        assert result.proof_source == "p25_5_skeletal_replacement"
        assert "octahydro" in result.name
        assert "cyclopenta[cd]indene" in result.name
        assert "cyclo[" not in result.name
        assert result.self_audit.coverage is not None
        assert not result.self_audit.coverage.unnamed_atoms
        assembly = next(step for step in reversed(result.decisions) if "name_token_spans" in step.data)
        hydrogen_tokens = [
            token
            for token in assembly.data["name_token_spans"]
            if token["grammar_role"] == "replacement_indicated_hydrogen" and token["text"] == "H"
        ]
        assert len(hydrogen_tokens) == indicated_count
        assert all(
            permuted.GetAtomWithIdx(atom).GetAtomicNum() == 6 for token in hydrogen_tokens for atom in token["atoms"]
        )
        names.add(result.name)
    assert len(names) == 1


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("smiles,indicated_count", CASES)
def test_every_atom_order_has_an_exact_opsin_roundtrip(smiles, indicated_count):
    graph = Chem.MolFromSmiles(smiles)
    for order in _orders(graph.GetNumAtoms()):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert result.parent_nomenclature == "skeletal_replacement_fusion"
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_roundtrip == check.canonical_original


@pytest.mark.parametrize("smiles,indicated_count", CASES)
@pytest.mark.parametrize("corruption", ["hydrogen", "bond", "charge"])
def test_joint_proof_rejects_corrupt_observed_state(smiles, indicated_count, corruption):
    mol, parent = _plan(Chem.MolFromSmiles(smiles))
    atom = next(iter(parent.atoms))
    if corruption == "hydrogen":
        mol.update_atom(atom, total_h_count=mol.atoms[atom].total_h_count + 1)
    elif corruption == "charge":
        mol.update_atom(atom, charge=1)
    else:
        edge = parent.replacement_fusion_state.derivative_state.bond_delta.hydrogenated_edges[0]
        mol.update_bond(mol.get_bond(*edge).idx, order=2)
    assert prove_replacement_state(mol, parent.fusion_plan) is None


@pytest.mark.parametrize("smiles,indicated_count", CASES)
def test_replacement_audit_rejects_an_inconsistent_hydrogen_balance(smiles, indicated_count):
    _, parent = _plan(Chem.MolFromSmiles(smiles))
    state = parent.replacement_fusion_state
    first, *rest = state.hydrogen_balances
    corrupted = replace(
        state, hydrogen_balances=(replace(first, observed_hydrogens=first.observed_hydrogens + 1), *rest)
    )
    assert not corrupted.audit_ok
    assert not replace(parent, replacement_fusion_state=corrupted).audit_ok


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_restored_chalcogen_capacity_is_not_oxygen_specific():
    graph = Chem.MolFromSmiles("C[C@H]1CC[C@@]2(C3(C)OCCO3)CC[C@H]3CON1[C@H]32")
    mol, parent = _plan(graph)
    for atom in parent.atoms:
        if mol.atoms[atom].symbol == "O":
            graph.GetAtomWithIdx(atom).SetAtomicNum(16)
    Chem.SanitizeMol(graph)
    Chem.AssignStereochemistry(graph, cleanIt=True, force=True)
    smiles = Chem.MolToSmiles(graph)
    names = set()
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True, verify_self=True)
        assert result.error is None
        assert result.parent_nomenclature == "skeletal_replacement_fusion"
        assert "thia" in result.name
        assert not result.self_audit.coverage.unnamed_atoms
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip
        names.add(result.name)
    assert len(names) == 1
