"""External pi consumption can leave suffix-owned nitrogen hydrogen."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import AuditStatus, FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _graph(heteroatom, external, chain_length):
    graph = Chem.RWMol()
    for symbol in (external, "C", "N", "C", heteroatom, "C", "C", "C", "O"):
        graph.AddAtom(Chem.Atom(symbol))
    for edge in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (3, 7), (7, 8), (1, 8)):
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in {(0, 1), (3, 4)} else Chem.BondType.SINGLE)
    last = 5
    for _ in range(chain_length):
        atom = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, atom, Chem.BondType.SINGLE)
        last = atom
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("heteroatom", ("C", "N"))
@pytest.mark.parametrize("external", ("O", "N"))
@pytest.mark.parametrize("chain_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_external_pi_consumption_cites_added_nh(heteroatom, external, chain_length, reverse):
    rd_mol = _graph(heteroatom, external, chain_length)
    order = list(range(rd_mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    state = result.plan.derivative_state
    added = {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids}
    assert order.index(2) in added
    assert not added.intersection(atom for operation in state.hydro_operations for atom in operation.atom_ids)
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("external", ("O", "N"))
@pytest.mark.parametrize("corruption", ("missing_added", "added_locant", "added_charge", "added_bond"))
def test_added_nh_requires_exact_operation_and_neutral_sigma_valence(external, corruption):
    mol = read_rdkit_mol(_graph("C", external, 0))
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    state = plan.derivative_state
    if corruption == "missing_added":
        state = replace(state, bond_delta=replace(state.bond_delta, added_hydrogen_operations=()))
    elif corruption == "added_locant":
        operation = state.added_hydrogen_operations[0]
        operations = (replace(operation, locants=("99",) + operation.locants[1:]),)
        state = replace(state, bond_delta=replace(state.bond_delta, added_hydrogen_operations=operations))
    elif corruption == "added_charge":
        mol.update_atom(2, charge=1)
    else:
        mol.update_bond(mol.get_bond(1, 2).idx, order=2)
    audit = audit_fusion_plan(
        mol,
        atoms,
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        derivative_state=state,
        mode="audited_pin",
    )
    assert audit.status is not AuditStatus.CONFIRMED
