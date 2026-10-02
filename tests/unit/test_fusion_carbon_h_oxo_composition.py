"""Carbon indicated H and a separately consumed carbonyl parent pi bond."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import _has_carbon_h_oxo_bond_consumption, audit_fusion_plan
from openclatura.fusion.model import AuditStatus, FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles
from openclatura.opsin_verify import verify_with_opsin

CASES = (
    ("C1Cc2ncccc2C1=O", "7H-cyclopenta[b]pyridin-5-one", "7", "5"),
    ("CC1Cc2ncccc2C1=O", "6-methyl-7H-cyclopenta[b]pyridin-5-one", "7", "5"),
    ("C1C(C)c2ncccc2C1=O", "7-methyl-7H-cyclopenta[b]pyridin-5-one", "7", "5"),
    ("C1Cc2cnccc2C1=O", "7H-cyclopenta[c]pyridin-5-one", "7", "5"),
    ("C1Cc2nccnc2C1=O", "5H-cyclopenta[b]pyrazin-7-one", "5", "7"),
    ("C1Cc2ncc(Cl)cc2C1=O", "3-chloro-7H-cyclopenta[b]pyridin-5-one", "7", "5"),
)


def _plan(smiles, mode=FusionMode.AUDITED_PIN):
    mol = read_smiles(smiles)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=mode)
    assert isinstance(result, FusionConfirmed)
    return mol, atoms, result.plan


@pytest.mark.parametrize("smiles,expected,h_locant,oxo_locant", CASES)
@pytest.mark.parametrize("order", ("original", "reversed", "shuffled", "kekule"))
def test_carbon_h_oxo_composition_is_graph_bound_and_order_invariant(smiles, expected, h_locant, oxo_locant, order):
    rd_mol = Chem.MolFromSmiles(smiles)
    indices = list(range(rd_mol.GetNumAtoms()))
    if order == "reversed":
        indices.reverse()
    elif order == "shuffled":
        random.Random(19).shuffle(indices)
    rd_mol = Chem.RenumberAtoms(rd_mol, indices)
    serialized = Chem.MolToSmiles(rd_mol, canonical=False, kekuleSmiles=order == "kekule")
    mol, atoms, plan = _plan(serialized)
    state = plan.derivative_state
    assert tuple(map(str, plan.indicated_hydrogens)) == (h_locant,)
    assert not state.hydro_operations
    assert not state.added_hydrogen_operations
    assert not state.intrinsic_hydro_operations
    assert not state.unsaturation_operations
    (oxo,) = state.oxo_operations
    assert oxo.locant == oxo_locant
    assert _has_carbon_h_oxo_bond_consumption(mol, atoms, state)
    assert mol.get_bond(oxo.parent_atom_id, oxo.oxygen_atom_id).idx == oxo.bond_id
    general = _plan(serialized, FusionMode.GENERAL)[2]
    assert general.derivative_state == state
    assert general.indicated_hydrogens == plan.indicated_hydrogens


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles,expected,h_locant,oxo_locant", CASES)
def test_carbon_h_oxo_grammar_and_public_names_roundtrip_exactly(smiles, expected, h_locant, oxo_locant):
    assert verify_with_opsin(expected, smiles, standardize_smiles=False).status == "matched"
    result = name_mol(Chem.MolFromSmiles(smiles), fusion_mode=FusionMode.AUDITED_PIN, verify_opsin=True)
    assert result.error is None
    assert result.opsin_check.status == "matched"


@pytest.mark.parametrize("corruption", ("missing_consumed_pi", "oxo_locant", "oxo_bond", "missing_h"))
def test_carbon_h_oxo_composition_rejects_corrupted_proofs(corruption):
    mol, atoms, plan = _plan(CASES[0][0])
    state = plan.derivative_state
    h = plan.indicated_hydrogens
    (oxo,) = state.oxo_operations
    if corruption == "missing_consumed_pi":
        assignment = replace(
            state.bond_delta.assignment,
            orders=tuple(
                (edge, 1 if oxo.parent_atom_id in edge else order) for edge, order in state.bond_delta.assignment.orders
            ),
        )
        state = replace(state, bond_delta=replace(state.bond_delta, assignment=assignment))
        assert not _has_carbon_h_oxo_bond_consumption(mol, atoms, state)
    elif corruption == "oxo_locant":
        state = replace(state, oxo_operations=(replace(oxo, locant="99"),))
    elif corruption == "oxo_bond":
        state = replace(state, oxo_operations=(replace(oxo, bond_id=-1),))
    else:
        h = ()
    result = audit_fusion_plan(
        mol,
        atoms,
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=h,
        derivative_state=state,
        mode=FusionMode.GENERAL,
    )
    assert result.status is AuditStatus.MISMATCH


def test_carbon_h_thioxo_is_not_promoted_by_the_oxo_composition_proof():
    mol = read_smiles("C1Cc2ncccc2C1=S")
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    assert not isinstance(plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN), FusionConfirmed)
