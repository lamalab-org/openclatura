"""External pi composition must not hydrogenate aromatic redistribution vertices."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.mancude import PiRedistribution, _single_site_parent_model, prove_pi_redistribution
from openclatura.fusion.model import AuditStatus, BondAssignment, FusionConfirmed, FusionMode
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
# The substituted N parents now compose directly; the carbon parent still
# requires alternating redistribution to exclude aromatic hydro endpoints.
REDISTRIBUTION_CASES = CASES[:1]
DIRECT_CASES = CASES[1:]


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
    arguments = dict(
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        derivative_state=state,
        charge_operations=plan.charge_operations,
        lambda_descriptors=plan.lambda_descriptors,
    )
    # The PIN capability gate may reject missing hydro before reconstruction.
    # Require production rejection there AND the precise corruption diagnosis
    # from the complete independent audit, without its optional PIN gate.
    pin = audit_fusion_plan(mol, atoms, **arguments, mode=FusionMode.AUDITED_PIN)
    result = audit_fusion_plan(mol, atoms, **arguments, mode=FusionMode.GENERAL)
    assert result.status is (AuditStatus.MISMATCH if result.errors else AuditStatus.CONFIRMED), result
    assert pin.confirmed == result.confirmed, pin
    assert bool(pin.errors) == bool(result.errors), pin
    return result.errors


def _observed_assignment(mol, atoms):
    return BondAssignment(
        tuple(
            sorted(
                (tuple(sorted((bond.u, bond.v))), bond.order)
                for bond in mol.bonds.values()
                if bond.u in atoms and bond.v in atoms
            )
        )
    )


@pytest.mark.parametrize("smiles", REDISTRIBUTION_CASES)
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
    observed = _observed_assignment(mol, atoms)
    assert proof.final_assignment == observed
    assert len(proof.hydrogenated_atom_ids) == 2 * (len(proof.removed_bond_ids) - len(proof.added_bond_ids))
    for operation in state.oxo_operations:
        assert operation.parent_atom_id not in hydro | added_h
        assert operation.bond_id not in proof.removed_bond_ids | proof.added_bond_ids
        for assignment in model.allowed_kekule_assignments:
            assert all(order == 1 for edge, order in assignment.orders if operation.parent_atom_id in edge)
    assert not _audit(mol, atoms, plan, state)


@pytest.mark.parametrize("smiles", DIRECT_CASES)
@pytest.mark.parametrize("order", ORDERS)
def test_direct_composition_replays_all_parent_bonds_without_aromatic_hydro(smiles, order):
    mol, atoms, plan = _plan(smiles, order)
    state = plan.derivative_state
    delta = state.bond_delta
    model = delta.composition_model
    assert model is not None
    assert delta.assignment in model.allowed_kekule_assignments
    assert state.pi_redistribution is None
    assert state.oxo_operations and state.added_hydrogen_operations
    assert delta.hydrogenated_edges
    assert not delta.additional_multiple_bond_ids
    assert not state.unsaturation_operations
    assert not state.intrinsic_hydro_operations
    hydro = {atom for operation in state.hydro_operations for atom in operation.atom_ids}
    added_h = {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids}
    aromatic = {atom for atom in atoms if mol.atoms[atom].is_aromatic}
    assert hydro == delta.hydrogenated_atom_ids
    assert not aromatic & (hydro | added_h)
    assert not hydro & added_h
    replayed = dict(delta.assignment.orders)
    for edge in delta.hydrogenated_edges:
        assert replayed[edge] == 2
        replayed[edge] = 1
    observed = _observed_assignment(mol, atoms)
    final_model = _single_site_parent_model(model, frozenset(hydro))
    assert BondAssignment(tuple(sorted(replayed.items()))) in final_model.allowed_kekule_assignments
    assert observed in final_model.allowed_kekule_assignments
    assert set(replayed) == set(dict(observed.orders))
    # Aromatic Kekule choices may differ after permutation, but every changed
    # bond must be aromatic and every atom must retain its exact pi occupancy.
    assert all(set(edge) <= aromatic for edge, order in observed.orders if replayed[edge] != order)
    for atom in atoms:
        assert sum(order - 1 for edge, order in replayed.items() if atom in edge) == sum(
            order - 1 for edge, order in observed.orders if atom in edge
        )
    assert len(hydro) == 2 * len(delta.hydrogenated_edges)
    locants = dict(plan.numbering.input_locant_maps[0])
    for operation in state.hydro_operations:
        assert operation.locants == tuple(str(locants[atom]) for atom in operation.atom_ids)
        assert set(operation.bond_ids) == {mol.get_bond(*edge).idx for edge in delta.hydrogenated_edges}
    for operation in state.oxo_operations:
        assert operation.parent_atom_id not in hydro | added_h
        assert all(
            order == 1
            for assignment in model.allowed_kekule_assignments
            for edge, order in assignment.orders
            if operation.parent_atom_id in edge
        )
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
        is None
    )
    assert not _audit(mol, atoms, plan, state)


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles", CASES)
@pytest.mark.parametrize("order", ORDERS)
def test_report_inputs_roundtrip_exactly_after_composition(smiles, order):
    result = name_mol(_permuted_mol(smiles, order), fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)
    assert result.error is None
    _, _, plan = _plan(smiles, order)
    # The fused system may be a substituent rather than the principal parent.
    assert plan.rendered_base_name.removesuffix("e") in result.name
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True)
    assert check.status == "matched", check
    assert check.canonical_original == check.canonical_roundtrip == canonical
    assert Chem.MolToSmiles(Chem.MolFromSmiles(check.opsin_smiles), canonical=True, isomericSmiles=True) == canonical


@pytest.mark.parametrize("smiles", CASES)
@pytest.mark.parametrize("corruption", ("missing_model", "model_assignments"))
def test_audit_rejects_corrupted_composition_domains(smiles, corruption):
    mol, atoms, plan = _plan(smiles)
    state = plan.derivative_state
    model = state.bond_delta.composition_model
    assert model is not None
    assert not _audit(mol, atoms, plan, state)
    corrupted = None if corruption == "missing_model" else replace(model, allowed_kekule_assignments=())
    state = replace(state, bond_delta=replace(state.bond_delta, composition_model=corrupted))
    assert any("selected parent bond delta" in error for error in _audit(mol, atoms, plan, state))


@pytest.mark.parametrize("smiles", REDISTRIBUTION_CASES)
@pytest.mark.parametrize("corruption", ("missing_redistribution", "aromatic_endpoint", "added_bonds"))
def test_audit_rejects_corrupted_redistribution_proofs(smiles, corruption):
    mol, atoms, plan = _plan(smiles)
    state = plan.derivative_state
    assert state.bond_delta.composition_model is not None
    assert state.pi_redistribution is not None
    assert not _audit(mol, atoms, plan, state)
    proof = state.pi_redistribution
    if corruption == "missing_redistribution":
        proof = None
    elif corruption == "aromatic_endpoint":
        aromatic = next(atom for atom in state.bond_delta.hydrogenated_atom_ids if mol.atoms[atom].is_aromatic)
        proof = replace(proof, hydrogenated_atom_ids=proof.hydrogenated_atom_ids | {aromatic})
    else:
        proof = replace(proof, added_bond_ids=frozenset())
    state = replace(state, pi_redistribution=proof)
    errors = _audit(mol, atoms, plan, state)
    assert any("pi redistribution" in error for error in errors), errors


@pytest.mark.parametrize("smiles", DIRECT_CASES)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "corruption",
    (
        "missing_hydro",
        "aromatic_endpoint",
        "hydro_bonds",
        "missing_added_h",
        "spurious_redistribution",
    ),
)
def test_audit_rejects_corrupted_direct_composition(smiles, order, corruption):
    mol, atoms, plan = _plan(smiles, order)
    state = plan.derivative_state
    delta = state.bond_delta
    assert state.pi_redistribution is None
    assert not _audit(mol, atoms, plan, state)
    expected_error = "hydro operation"
    if corruption == "missing_hydro":
        state = replace(state, hydro_operations=())
    elif corruption in {"aromatic_endpoint", "hydro_bonds"}:
        operation = state.hydro_operations[0]
        if corruption == "aromatic_endpoint":
            aromatic = next(atom for atom in atoms if mol.atoms[atom].is_aromatic)
            endpoints = (aromatic, *operation.atom_ids[1:])
            locants = plan.numbering.string_input_locant_maps()[0]
            operation = replace(operation, atom_ids=endpoints, locants=tuple(locants[atom] for atom in endpoints))
        else:
            operation = replace(operation, bond_ids=())
        state = replace(state, hydro_operations=(operation,))
    elif corruption == "missing_added_h":
        state = replace(state, bond_delta=replace(delta, added_hydrogen_operations=()))
        expected_error = "selected parent bond delta"
    else:
        # A direct reduction is not an alternating redistribution witness,
        # even when its net endpoints and final assignment are correct.
        proof = PiRedistribution(
            frozenset(mol.get_bond(*edge).idx for edge in delta.hydrogenated_edges),
            frozenset(),
            delta.hydrogenated_atom_ids,
            _observed_assignment(mol, atoms),
        )
        state = replace(state, pi_redistribution=proof)
        expected_error = "pi redistribution"
    errors = _audit(mol, atoms, plan, state)
    assert any(expected_error in error for error in errors), errors
