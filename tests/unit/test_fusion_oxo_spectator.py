"""An unchanged oxo site must not block proved remote pi redistribution."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol, opsin_available
from openclatura.fusion.audit import _audit_derivative_state
from openclatura.fusion.mancude import _single_site_parent_model, prove_pi_redistribution
from openclatura.fusion.model import BondAssignment, FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles
from openclatura.opsin_verify import verify_with_opsin

SMILES = "O=C1OC2=C(COC2)O1"
EXPECTED = "4,6-dihydrofuro[3,4-d][1,3]dioxol-2-one"


def _case(smiles=SMILES):
    mol = read_smiles(smiles)
    atoms = frozenset(atom for atom in mol.atoms if len(mol.get_neighbors(atom)) > 1)
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    return mol, atoms, result.plan


@pytest.mark.parametrize("smiles", (SMILES, "O=c1oc2c(o1)COC2"))
def test_unchanged_oxo_site_leaves_a_proved_remote_hydro_pair(smiles):
    mol, atoms, plan = _case(smiles)
    locants = dict(plan.numbering.input_locant_maps[0])
    by_locant = {str(locant): atom for atom, locant in locants.items()}
    state = plan.derivative_state
    assert len(state.oxo_operations) == 1
    oxo = state.oxo_operations[0]
    assert str(oxo.locant) == "2"
    assert mol.get_bond(oxo.parent_atom_id, oxo.oxygen_atom_id).order == 2
    assignment = dict(state.bond_delta.assignment.orders)
    assert state.bond_delta.assignment in plan.bond_model.allowed_kekule_assignments
    oxo_edges = [edge for edge in assignment if oxo.parent_atom_id in edge]
    assert len(oxo_edges) == 2
    assert all(assignment[edge] == mol.get_bond(*edge).order == 1 for edge in oxo_edges)

    sites = frozenset(by_locant[locant] for locant in ("4", "6"))
    assert oxo.parent_atom_id not in sites
    assert all(mol.atoms[atom].symbol == "C" and mol.atoms[atom].total_h_count == 2 for atom in sites)
    implied = _single_site_parent_model(plan.bond_model, sites)
    observed = BondAssignment(tuple(sorted((edge, mol.get_bond(*edge).order) for edge in assignment)))
    assert observed in implied.allowed_kekule_assignments
    assert implied.maximum_non_cumulative_double_bonds == plan.bond_model.maximum_non_cumulative_double_bonds - 1
    shared = tuple(sorted((by_locant["3a"], by_locant["6a"])))
    assert dict(observed.orders)[shared] == 2
    assert state.pi_redistribution is not None
    assert state.pi_redistribution.final_assignment == observed
    assert state.pi_redistribution.hydrogenated_atom_ids == sites
    assert state.hydro_operations[0].locants == ("4", "6")
    assert oxo.bond_id not in state.pi_redistribution.removed_bond_ids | state.pi_redistribution.added_bond_ids
    assert prove_pi_redistribution(mol, atoms, plan.bond_model, state.bond_delta) is None


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
def test_oxo_spectator_expected_name_has_raw_exact_opsin_identity():
    check = verify_with_opsin(EXPECTED, SMILES, standardize_smiles=False)
    assert check.ok, check
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(SMILES), canonical=True, isomericSmiles=True)
    assert check.canonical_original == check.canonical_roundtrip == expected
    assert Chem.MolToSmiles(Chem.MolFromSmiles(check.opsin_smiles), canonical=True, isomericSmiles=True) == expected


def test_oxo_spectator_production_uses_the_proved_endpoints():
    result = name_mol(Chem.MolFromSmiles(SMILES), fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)
    assert result.parent_nomenclature == "systematic_fusion"
    assert result.name == EXPECTED


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
def test_oxo_spectator_production_atom_reversal_has_raw_exact_identity():
    mol = Chem.MolFromSmiles(SMILES)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(
            Chem.RenumberAtoms(mol, order),
            fusion_mode=FusionMode.AUDITED_PIN,
            include_trace=True,
        )
        assert result.name == EXPECTED
        check = verify_with_opsin(result.name, SMILES, standardize_smiles=False)
        assert check.ok, check
        assert check.canonical_original == check.canonical_roundtrip == canonical


@pytest.mark.parametrize("corruption", ("duplicate", "oxygen", "bond", "charged", "nonterminal", "thioxo"))
def test_spectator_oxo_requires_a_valid_typed_terminal_group(corruption):
    mol, atoms, plan = _case()
    state = plan.derivative_state
    operation = state.oxo_operations[0]
    operations = state.oxo_operations
    if corruption == "duplicate":
        operations *= 2
    elif corruption == "oxygen":
        operations = (replace(operation, oxygen_atom_id=operation.parent_atom_id),)
    elif corruption == "bond":
        operations = (replace(operation, bond_id=max(mol.bonds) + 1),)
    elif corruption == "charged":
        mol.update_atom(operation.oxygen_atom_id, charge=-1)
    elif corruption == "thioxo":
        mol.update_atom(operation.oxygen_atom_id, symbol="S")
    else:
        branch = max(mol.atoms) + 1
        mol.add_atom("C", idx=branch, total_h_count=3)
        mol.add_bond(operation.oxygen_atom_id, branch, idx=max(mol.bonds) + 1)
    assert (
        prove_pi_redistribution(
            mol,
            atoms,
            plan.bond_model,
            state.bond_delta,
            oxo_operations=operations,
        )
        is None
    )


def test_oxo_cannot_be_a_pi_changing_site_or_indicated_h_site():
    mol, atoms, plan = _case()
    state = plan.derivative_state
    parent = state.oxo_operations[0].parent_atom_id
    assert (
        prove_pi_redistribution(
            mol,
            atoms,
            plan.bond_model,
            state.bond_delta,
            indicated_hydrogen_atom_ids={parent},
            oxo_operations=state.oxo_operations,
        )
        is None
    )
    orders = dict(state.bond_delta.assignment.orders)
    edge = next(edge for edge in orders if parent in edge)
    orders[edge] = 2
    assignment = BondAssignment(tuple(sorted(orders.items())))
    model = replace(
        plan.bond_model,
        allowed_kekule_assignments=(assignment,),
        required_single_bonds=plan.bond_model.required_single_bonds - {edge},
        pi_eligible_edges=plan.bond_model.pi_eligible_edges | {edge},
        maximum_non_cumulative_double_bonds=plan.bond_model.maximum_non_cumulative_double_bonds + 1,
    )
    delta = replace(state.bond_delta, assignment=assignment)
    assert prove_pi_redistribution(mol, atoms, model, delta, oxo_operations=state.oxo_operations) is None


@pytest.mark.parametrize("corruption", ("missing_oxo", "oxo_locant", "missing_redistribution", "hydro_locant"))
def test_oxo_redistribution_audit_rejects_corrupted_operations(corruption):
    mol, atoms, plan = _case()
    state = plan.derivative_state
    if corruption == "missing_oxo":
        state = replace(state, oxo_operations=())
    elif corruption == "oxo_locant":
        state = replace(state, oxo_operations=(replace(state.oxo_operations[0], locant="4"),))
    elif corruption == "missing_redistribution":
        state = replace(state, pi_redistribution=None)
    else:
        state = replace(state, hydro_operations=(replace(state.hydro_operations[0], locants=("3a", "4")),))
    errors = []
    _audit_derivative_state(mol, atoms, plan.numbering, plan.bond_model, plan.indicated_hydrogens, state, errors)
    assert errors
