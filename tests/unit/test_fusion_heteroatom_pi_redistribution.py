"""Neutral pi spectators and separately proved amine hydro endpoints."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol, opsin_available
from openclatura.fusion.audit import _audit_derivative_state
from openclatura.fusion.mancude import parent_derivative_state, prove_pi_redistribution
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol, read_smiles
from openclatura.opsin_verify import verify_with_opsin

SPECTATOR = "C1OCC2=NC=CC=C12"
ENDPOINT = "C1N=CNC2=C1N=CO2"


def _case(smiles):
    mol = read_smiles(smiles)
    result = plan_fusion_parent(mol, frozenset(mol.atoms), mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    return mol, result.plan


def test_nitrogen_spectator_exchanges_pi_bonds_without_losing_occupancy():
    mol, plan = _case(SPECTATOR)
    state = plan.derivative_state
    proof = state.pi_redistribution
    assert proof is not None
    nitrogen = next(atom.idx for atom in mol.atoms.values() if atom.symbol == "N")
    before = {edge: order for edge, order in state.bond_delta.assignment.orders if nitrogen in edge}
    after = {edge: order for edge, order in proof.final_assignment.orders if nitrogen in edge}
    assert before != after
    assert sum(order - 1 for order in before.values()) == sum(order - 1 for order in after.values()) == 1
    assert nitrogen not in proof.hydrogenated_atom_ids
    assert state.hydro_operations[0].locants == ("5", "7")
    assert not state.unsaturation_operations


def test_neutral_amine_endpoint_loses_exactly_one_pi_bond():
    mol, plan = _case(ENDPOINT)
    state = plan.derivative_state
    proof = state.pi_redistribution
    assert proof is not None
    assert state.hydro_operations[0].locants == ("4", "7")
    nitrogen = next(atom for atom in proof.hydrogenated_atom_ids if mol.atoms[atom].symbol == "N")
    assert len(proof.hydrogenated_atom_ids) == 2
    assert mol.atoms[nitrogen].total_h_count == 1
    assert not mol.atoms[nitrogen].is_aromatic
    assert sum(order - 1 for edge, order in state.bond_delta.assignment.orders if nitrogen in edge) == 1
    assert all(order == 1 for edge, order in proof.final_assignment.orders if nitrogen in edge)
    assert not plan.indicated_hydrogens
    errors = []
    _audit_derivative_state(
        mol,
        frozenset(mol.atoms),
        plan.numbering,
        plan.bond_model,
        (),
        state,
        errors,
    )
    assert not errors


def test_external_sigma_ligand_can_replace_endpoint_nh():
    mol, plan = _case(ENDPOINT)
    state = plan.derivative_state
    nitrogen = next(atom for atom in state.pi_redistribution.hydrogenated_atom_ids if mol.atoms[atom].symbol == "N")
    substituted = Chem.RWMol(Chem.MolFromSmiles(ENDPOINT))
    methyl = substituted.AddAtom(Chem.Atom("C"))
    substituted.AddBond(nitrogen, methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(substituted)
    modified = read_rdkit_mol(substituted)
    assert modified.atoms[nitrogen].total_h_count == 0
    locants = dict(plan.numbering.input_locant_maps[0])
    result = parent_derivative_state(modified, frozenset(locants), plan.bond_model, locants)
    assert result is not None and result.pi_redistribution is not None
    assert result.pi_redistribution.hydrogenated_atom_ids == state.pi_redistribution.hydrogenated_atom_ids


@pytest.mark.parametrize("change", ("charged", "aromatic", "oxygen", "cited_h"))
def test_nitrogen_endpoint_gate_does_not_admit_other_endpoint_classes(change):
    mol, plan = _case(ENDPOINT)
    state = plan.derivative_state
    nitrogen = next(atom for atom in state.pi_redistribution.hydrogenated_atom_ids if mol.atoms[atom].symbol == "N")
    cited = set()
    if change == "charged":
        mol.update_atom(nitrogen, charge=1, total_h_count=2)
    elif change == "aromatic":
        mol.update_atom(nitrogen, is_aromatic=True)
    elif change == "oxygen":
        mol.update_atom(nitrogen, symbol="O", total_h_count=0)
    else:
        cited.add(nitrogen)
    assert (
        prove_pi_redistribution(
            mol,
            frozenset(mol.atoms),
            plan.bond_model,
            state.bond_delta,
            indicated_hydrogen_atom_ids=cited,
        )
        is None
    )


@pytest.mark.parametrize("order", (1, 2))
def test_spectator_relocation_still_respects_fixed_bonds(order):
    mol, plan = _case(SPECTATOR)
    state = plan.derivative_state
    nitrogen = next(atom.idx for atom in mol.atoms.values() if atom.symbol == "N")
    edge = next(edge for edge, value in state.bond_delta.assignment.orders if nitrogen in edge and value == order)
    model = replace(
        plan.bond_model,
        allowed_kekule_assignments=(state.bond_delta.assignment,),
        pi_eligible_edges=plan.bond_model.pi_eligible_edges - {edge},
        required_single_bonds=plan.bond_model.required_single_bonds | ({edge} if order == 1 else set()),
        required_double_bonds=plan.bond_model.required_double_bonds | ({edge} if order == 2 else set()),
    )
    assert prove_pi_redistribution(mol, frozenset(mol.atoms), model, state.bond_delta) is None


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    "smiles,expected",
    (
        (SPECTATOR, "5,7-dihydrofuro[3,4-b]pyridine"),
        (ENDPOINT, "4,7-dihydrooxazolo[5,4-d]pyrimidine"),
        ("C1N=CN(C)C2=C1N=CO2", "4-methyl-4,7-dihydrooxazolo[5,4-d]pyrimidine"),
    ),
)
def test_heteroatom_redistribution_has_raw_exact_opsin_identity(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(
            Chem.RenumberAtoms(mol, order),
            fusion_mode=FusionMode.AUDITED_PIN,
            include_trace=True,
        )
        assert result.name == expected
        assert result.parent_nomenclature == "systematic_fusion"
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.ok, check
        assert check.canonical_original == check.canonical_roundtrip == canonical
        assert (
            Chem.MolToSmiles(Chem.MolFromSmiles(check.opsin_smiles), canonical=True, isomericSmiles=True) == canonical
        )
