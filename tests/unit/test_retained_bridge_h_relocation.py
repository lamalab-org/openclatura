"""Retained matching H locations stay aligned with wrapper names and models."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.wrappers import plan_bridged_fusion_wrapper
from openclatura.graph_io import read_rdkit_mol


def _bridged_nitrogen_parent(ligand_length, state="unsaturated"):
    graph = Chem.RWMol()
    for symbol in ("C",) * 8 + ("N",):
        graph.AddAtom(Chem.Atom(symbol))
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 0),
        (7, 8),
        (8, 1),
        (8, 4),
    )
    doubles = {(2, 3), (4, 5), (6, 7)}
    if state == "hydrogenated":
        doubles = {(2, 3), (5, 6)}
    elif state == "adjacent_bridge":
        edges = tuple(edge for edge in edges if edge != (7, 0)) + ((4, 0),)
        doubles = {(2, 3), (5, 6)}
    for u, v in edges:
        graph.AddBond(u, v, Chem.BondType.DOUBLE if (u, v) in doubles else Chem.BondType.SINGLE)
    last = 2
    for _ in range(ligand_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("ligand_length", (0, 1, 2))
@pytest.mark.parametrize("ordering", ("original", "reverse", "kekule"))
def test_retained_wrapper_consumes_matched_h_metadata(ligand_length, ordering):
    rd_mol = _bridged_nitrogen_parent(ligand_length)
    if ordering == "reverse":
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    elif ordering == "kekule":
        Chem.Kekulize(rd_mol, clearAromaticFlags=True)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    plan = plan_bridged_fusion_wrapper(mol, atoms, mode="audited_pin")
    assert plan is not None
    assert plan.parent.name == "3H-pyrrolizine"
    metadata = plan.parent.hydride.hydride_metadata
    assert metadata.default_indicated_h == ("3",)
    assert metadata.derivative_stem == "3H-pyrrolizin"
    assert metadata.inherent_saturated_locants == ("3",)
    assert metadata.relocated_indicated_h
    hydrogen_atom = next(atom for atom, locant in plan.parent.selected_locant_map if locant == "3")
    assert all(
        order == 1
        for assignment in plan.parent.selected_bond_model.allowed_kekule_assignments
        for edge, order in assignment.orders
        if hydrogen_atom in edge
    )
    assert not plan.derivative_state.hydro_operations
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "bridged_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("state", ("hydrogenated", "adjacent_bridge"))
@pytest.mark.parametrize("ligand_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_retained_h_relocation_preserves_neutral_junction_donor(state, ligand_length, reverse):
    rd_mol = _bridged_nitrogen_parent(ligand_length, state)
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    plan = plan_bridged_fusion_wrapper(mol, atoms, mode="audited_pin")
    assert plan is not None
    nitrogen = next(atom for atom in atoms if mol.atoms[atom].symbol == "N")
    assert all(
        order == 1
        for assignment in plan.parent.selected_bond_model.allowed_kekule_assignments
        for edge, order in assignment.orders
        if nitrogen in edge
    )
    assert nitrogen not in {atom for operation in plan.derivative_state.hydro_operations for atom in operation.atom_ids}
    assert sum(len(operation.atom_ids) for operation in plan.derivative_state.hydro_operations) == 2
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "bridged_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
