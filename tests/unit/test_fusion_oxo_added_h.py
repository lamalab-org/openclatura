"""Oxo substitution can relocate a parent pi bond and leave carbon added H."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import AuditStatus, FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _fused_oxo_graph(heteroatom, alkyl_length, *, hydrogenated=False, external="O", ligand_length=0):
    graph = Chem.RWMol()
    for symbol in ("C", heteroatom, "C", "C", "C", "C", "C", "N", external):
        graph.AddAtom(Chem.Atom(symbol))
    for u, v, order in (
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 2),
        (4, 0, 1),
        (4, 5, 1),
        (5, 6, 2),
        (6, 7, 1),
        (7, 3, 1),
        (0, 8, 2),
    ):
        if hydrogenated and (u, v) == (5, 6):
            order = 1
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    for last, length in ((7, alkyl_length), (8, ligand_length)):
        for _ in range(length):
            carbon = graph.AddAtom(Chem.Atom("C"))
            graph.AddBond(last, carbon, Chem.BondType.SINGLE)
            last = carbon
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("heteroatom", ("O", "N"))
@pytest.mark.parametrize("alkyl_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_oxo_and_donor_constraints_emit_added_h_not_junction_hydro(heteroatom, alkyl_length, reverse):
    rd_mol = _fused_oxo_graph(heteroatom, alkyl_length)
    order = list(range(rd_mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    state = result.plan.derivative_state
    assert not state.hydro_operations
    assert not state.unsaturation_operations
    assert len(state.oxo_operations) == 1
    assert {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids} == {order.index(2)}
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("heteroatom", ("O", "N"))
@pytest.mark.parametrize("alkyl_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_oxo_added_h_composes_with_partial_hydrogenation(heteroatom, alkyl_length, reverse):
    rd_mol = _fused_oxo_graph(heteroatom, alkyl_length, hydrogenated=True)
    order = list(range(rd_mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    plan = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(plan, FusionConfirmed), plan
    state = plan.plan.derivative_state
    assert state.hydro_operations
    assert not state.unsaturation_operations
    assert len(state.oxo_operations) == 1
    added = {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids}
    hydro = {atom for operation in state.hydro_operations for atom in operation.atom_ids}
    assert added
    assert not added & hydro
    assert order.index(0) not in added | hydro
    assert all(mol.atoms[atom].symbol == "C" for atom in added)
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("external", ("O", "N"))
@pytest.mark.parametrize("corruption", ("missing_h", "donor_pi", "overlapping_hydro", "external_bond"))
def test_multiple_nh_partial_oxo_rejects_corrupted_operations(corruption, external):
    mol = read_rdkit_mol(_fused_oxo_graph("N", 0, hydrogenated=True, external=external))
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    state = plan.derivative_state
    hydrogens = plan.indicated_hydrogens
    if corruption == "missing_h":
        hydrogens = hydrogens[1:]
    elif corruption == "donor_pi":
        assignment = replace(
            state.bond_delta.assignment,
            orders=tuple((edge, 2 if 1 in edge else order) for edge, order in state.bond_delta.assignment.orders),
        )
        state = replace(state, bond_delta=replace(state.bond_delta, assignment=assignment))
    elif corruption == "overlapping_hydro":
        state = replace(state, hydro_operations=state.hydro_operations + state.added_hydrogen_operations)
    else:
        field = "oxo_operations" if external == "O" else "imino_operations"
        state = replace(state, **{field: (replace(getattr(state, field)[0], bond_id=-1),)})
    audit = audit_fusion_plan(
        mol,
        atoms,
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=hydrogens,
        derivative_state=state,
        mode="audited_pin",
    )
    assert audit.status is not AuditStatus.CONFIRMED


@pytest.mark.parametrize("heteroatom", ("O", "N"))
@pytest.mark.parametrize("hydrogenated", (False, True))
@pytest.mark.parametrize("ligand_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_imino_uses_external_pi_composition_without_losing_n_ligands(heteroatom, hydrogenated, ligand_length, reverse):
    rd_mol = _fused_oxo_graph(heteroatom, 1, hydrogenated=hydrogenated, external="N", ligand_length=ligand_length)
    order = list(range(rd_mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    state = result.plan.derivative_state
    assert not state.oxo_operations
    (imino,) = state.imino_operations
    assert imino.parent_atom_id == order.index(0)
    assert imino.nitrogen_atom_id == order.index(8)
    assert mol.get_bond(imino.parent_atom_id, imino.nitrogen_atom_id).idx == imino.bond_id
    assert not state.unsaturation_operations
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("external", ("O", "N"))
@pytest.mark.parametrize("double_edge", ((5, 6), (6, 7)))
@pytest.mark.parametrize("reverse", (False, True))
def test_cited_donor_zero_edge_delta_still_cites_residual_carbon_h(external, double_edge, reverse):
    graph = Chem.RWMol()
    for symbol in (external, "C", "N", "C", "C", "C", "C", "C", "O"):
        graph.AddAtom(Chem.Atom(symbol))
    for edge in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (3, 7), (4, 8), (1, 8)):
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in {(0, 1), (3, 4), double_edge} else Chem.BondType.SINGLE)
    rd_mol = graph.GetMol()
    Chem.SanitizeMol(rd_mol)
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    state = result.plan.derivative_state
    assert not state.hydro_operations
    assert not state.unsaturation_operations
    assert state.added_hydrogen_operations
    assert len(state.external_pi_operations) == 1
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
