"""Engine and operation proofs for intrinsic carbon H and spiro oxo added H."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.mancude import _spiro_carbon_sites
from openclatura.fusion.model import AuditStatus, FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles

CARBON_CASES = (
    (
        "CC(=O)OC1Oc2ccc(C)cc2-c2oc(=O)c([Se]c3ccccc3)cc21",
        "9-methyl-2-oxo-3-(phenylselanyl)-2,5-dihydropyrano[5,6-c]benzo[e]pyran-5-yl acetate",
        "intrinsic_hydro_operations",
        ("2", "5"),
    ),
    (
        "CC1=C2CC3C(C)(C=CC(=O)C34CO4)CC2OC1=O",
        "3,8a-dimethyl-4,4a,8a,9-tetrahydrospiro[benzo[f]1-benzofuran-5,2'-oxirane]-2,6(9aH)-dione",
        "added_hydrogen_operations",
        ("9a",),
    ),
)


def _plan(smiles, mode=FusionMode.GENERAL):
    mol = read_smiles(smiles)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=mode)
    assert isinstance(result, FusionConfirmed)
    return mol, atoms, result.plan


@pytest.mark.parametrize("smiles,expected,field,locants", CARBON_CASES)
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("mode", (FusionMode.GENERAL, FusionMode.AUDITED_PIN))
def test_carbon_composition_is_generated_and_atom_order_invariant(smiles, expected, field, locants, reverse, mode):
    rd_mol = Chem.MolFromSmiles(smiles)
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    result = name_mol(rd_mol, fusion_mode=mode, verify_opsin=opsin_available(), include_trace=True)
    assert result.name == expected
    assert result.error is None
    if opsin_available():
        assert result.opsin_check.status == "matched"
    _, _, plan = _plan(Chem.MolToSmiles(rd_mol, canonical=False), mode)
    state = plan.derivative_state
    (operation,) = getattr(state, field)
    assert operation.locants == locants
    assert not state.unsaturation_operations
    if field == "added_hydrogen_operations":
        assert state.hydro_operations[0].locants == ("4", "4a", "8a", "9")
        assert not set(operation.atom_ids).intersection(state.hydro_operations[0].atom_ids)
        selected = next(
            step for step in result.decisions if step.decision == "selected audited systematic fusion parent"
        )
        assert selected.data["derivative_operations"]["added_hydrogen"][0]["locants"] == ["9a"]
    else:
        assert not state.hydro_operations
        assert set(operation.atom_ids).intersection(op.parent_atom_id for op in state.oxo_operations)


@pytest.mark.parametrize("smiles,expected,field,locants", CARBON_CASES)
@pytest.mark.parametrize("binding", ("locants", "atom_ids", "bond_ids"))
def test_carbon_operation_bindings_are_independently_audited(smiles, expected, field, locants, binding):
    mol, atoms, plan = _plan(smiles)
    delta = plan.derivative_state.bond_delta
    (operation,) = getattr(delta, field)
    original = getattr(operation, binding)
    corrupted_value = original[:-1] if original else (999,)
    corrupted = replace(operation, **{binding: corrupted_value})
    state = replace(plan.derivative_state, bond_delta=replace(delta, **{field: (corrupted,)}))
    result = audit_fusion_plan(
        mol,
        atoms,
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        charge_operations=plan.charge_operations,
        derivative_state=state,
        mode=FusionMode.GENERAL,
    )
    assert result.status is AuditStatus.MISMATCH
    assert "typed derivative state does not carry the selected parent bond delta" in result.errors


@pytest.mark.parametrize(
    "smiles",
    (
        "CCC1=C2CC3C(C)(C=CC(=O)C34CO4)CC2OC1=O",
        "CC1=C2CC3C(CC)(C=CC(=O)C34CO4)CC2OC1=O",
        "CCC(=O)OC1Oc2ccc(C)cc2-c2oc(=O)c([Se]c3ccccc3)cc21",
        "CC(=O)OC1Oc2ccc(CC)cc2-c2oc(=O)c([Se]c3ccccc3)cc21",
    ),
)
def test_carbon_composition_substituent_variants_round_trip(smiles):
    result = name_mol(Chem.MolFromSmiles(smiles), fusion_mode=FusionMode.GENERAL, verify_opsin=opsin_available())
    assert result.error is None
    if opsin_available():
        assert result.opsin_check.status == "matched"
    _, _, plan = _plan(smiles)
    assert plan.derivative_state.added_hydrogen_operations or plan.derivative_state.intrinsic_hydro_operations


def test_disconnected_external_substituents_do_not_prove_a_spiro_constraint():
    mol = read_smiles("CC1(CO)CCCCC1")
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    assert not _spiro_carbon_sites(mol, frozenset(atoms))


def test_external_bridge_between_parent_sites_is_not_a_spiro_constraint():
    mol = read_smiles("C1CC2CCC1C2")
    assert not _spiro_carbon_sites(mol, frozenset({0, 1, 2, 3, 4, 5}))
