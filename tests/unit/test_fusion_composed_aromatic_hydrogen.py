"""External pi composition must not hydrogenate aromatic redistribution vertices."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import _audit_derivative_state
from openclatura.fusion.mancude import prove_pi_redistribution
from openclatura.fusion.model import BondAssignment, FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles
from openclatura.opsin_verify import verify_with_opsin

# Preserve the report's original SMILES, including specified stereochemistry.
CASES = (
    pytest.param(
        "COc1ccc(-c2c3c(cc4c2C(=O)CC(C)(C)C4)CN(S(=O)(=O)c2ccc(C)cc2)C3)cc1",
        id="pubchem-100",
    ),
    pytest.param(
        "CNC(=O)c1ccc(Nc2ncc3c(n2)N(CC2CCC2)CC(C)(C)C(=O)N3C)c(OC)c1",
        id="pubchem-2806",
    ),
    pytest.param(
        "CCN1C(=O)[C@@H](N)[C@@H](c2ccc(F)cc2)c2c(C)nn(C3CCCCC3)c21",
        id="pubchem-4023",
    ),
)
ORDERS = ("original", "reversed", "shuffled")


def _permuted_mol(smiles, order):
    mol = Chem.MolFromSmiles(smiles)
    indices = list(range(mol.GetNumAtoms()))
    if order == "reversed":
        indices.reverse()
    elif order == "shuffled":
        random.Random(19).shuffle(indices)
    return Chem.RenumberAtoms(mol, indices)


def _plan(smiles, order="original"):
    serialized = Chem.MolToSmiles(_permuted_mol(smiles, order), canonical=False, isomericSmiles=True)
    mol = read_smiles(serialized)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    assert result.plan.audit.confirmed
    return mol, atoms, result.plan


def _audit(mol, atoms, plan, state):
    errors = []
    _audit_derivative_state(mol, atoms, plan.numbering, plan.bond_model, plan.indicated_hydrogens, state, errors)
    return errors


@pytest.mark.parametrize("smiles", CASES)
@pytest.mark.parametrize("order", ORDERS)
def test_composed_proof_excludes_aromatic_hydro_endpoints(smiles, order):
    mol, atoms, plan = _plan(smiles, order)
    state = plan.derivative_state
    delta = state.bond_delta
    model = delta.composition_model
    proof = state.pi_redistribution
    assert model is not None
    assert proof is not None
    assert delta.assignment in model.allowed_kekule_assignments
    assert state.oxo_operations
    assert state.added_hydrogen_operations

    aromatic = {atom for atom in atoms if mol.atoms[atom].is_aromatic}
    assert aromatic & delta.hydrogenated_atom_ids
    assert proof.hydrogenated_atom_ids
    assert not aromatic & proof.hydrogenated_atom_ids
    hydro = {atom for operation in state.hydro_operations for atom in operation.atom_ids}
    added_h = {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids}
    assert hydro == proof.hydrogenated_atom_ids
    assert not aromatic & (hydro | added_h)
    assert not hydro & added_h
    locants = dict(plan.numbering.input_locant_maps[0])
    for operation in state.hydro_operations:
        assert operation.locants == tuple(str(locants[atom]) for atom in operation.atom_ids)

    assert (
        prove_pi_redistribution(
            mol,
            atoms,
            plan.bond_model,
            delta,
            indicated_hydrogen_atom_ids={
                atom for atom, locant in locants.items() if locant in plan.indicated_hydrogens
            },
            oxo_operations=state.oxo_operations,
        )
        == proof
    )
    observed = BondAssignment(tuple(sorted((edge, mol.get_bond(*edge).order) for edge, _ in delta.assignment.orders)))
    assert proof.final_assignment == observed
    assert len(proof.hydrogenated_atom_ids) == 2 * (len(proof.removed_bond_ids) - len(proof.added_bond_ids))
    for operation in state.oxo_operations:
        assert operation.parent_atom_id not in hydro | added_h
        assert operation.bond_id not in proof.removed_bond_ids | proof.added_bond_ids
        for assignment in model.allowed_kekule_assignments:
            assert all(order == 1 for edge, order in assignment.orders if operation.parent_atom_id in edge)
    assert not _audit(mol, atoms, plan, state)


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles", CASES)
@pytest.mark.parametrize("order", ORDERS)
def test_report_inputs_roundtrip_exactly_after_composition(smiles, order):
    result = name_mol(_permuted_mol(smiles, order), fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True)
    assert check.status == "matched", check
    assert check.canonical_original == check.canonical_roundtrip == canonical
    assert Chem.MolToSmiles(Chem.MolFromSmiles(check.opsin_smiles), canonical=True, isomericSmiles=True) == canonical


@pytest.mark.parametrize("smiles", CASES)
@pytest.mark.parametrize(
    "corruption",
    ("missing_model", "model_assignments", "missing_redistribution", "aromatic_endpoint", "added_bonds"),
)
def test_audit_rejects_corrupted_composition_proofs(smiles, corruption):
    mol, atoms, plan = _plan(smiles)
    state = plan.derivative_state
    assert state.bond_delta.composition_model is not None
    assert state.pi_redistribution is not None
    assert not _audit(mol, atoms, plan, state)
    if corruption in {"missing_model", "model_assignments"}:
        model = state.bond_delta.composition_model
        corrupted = None if corruption == "missing_model" else replace(model, allowed_kekule_assignments=())
        state = replace(state, bond_delta=replace(state.bond_delta, composition_model=corrupted))
        expected_error = "selected parent bond delta"
    else:
        proof = state.pi_redistribution
        if corruption == "missing_redistribution":
            proof = None
        elif corruption == "aromatic_endpoint":
            aromatic = next(atom for atom in state.bond_delta.hydrogenated_atom_ids if mol.atoms[atom].is_aromatic)
            proof = replace(proof, hydrogenated_atom_ids=proof.hydrogenated_atom_ids | {aromatic})
        else:
            proof = replace(proof, added_bond_ids=frozenset())
        state = replace(state, pi_redistribution=proof)
        expected_error = "pi redistribution"
    errors = _audit(mol, atoms, plan, state)
    assert any(expected_error in error for error in errors), errors
