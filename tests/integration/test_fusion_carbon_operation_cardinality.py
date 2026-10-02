"""Graph-built probes of carbon-H composition beyond the original witnesses."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import AuditStatus, FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles


def _spiro_variant(change):
    mol = Chem.RWMol(Chem.MolFromSmiles("CC1=C2CC3C(C)(C=CC(=O)C34CO4)CC2OC1=O"))
    if change == "extra_oxo":
        oxygen = mol.AddAtom(Chem.Atom("O"))
        mol.AddBond(3, oxygen, Chem.BondType.DOUBLE)
    else:
        # Move the external oxirane from parent position 5 to position 8 or 9.
        for atom in mol.GetAtoms():
            atom.SetIntProp("original_index", atom.GetIdx())
        mol.RemoveAtom(13)
        mol.RemoveAtom(12)
        original = 7 if change == "spiro_at_8" else 14
        target = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIntProp("original_index") == original)
        for bond in mol.GetAtomWithIdx(target).GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
        carbon = mol.AddAtom(Chem.Atom("C"))
        oxygen = mol.AddAtom(Chem.Atom("O"))
        for first, second in ((target, carbon), (carbon, oxygen), (oxygen, target)):
            mol.AddBond(first, second, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    return mol


def _annellated_carbon_pair(edge):
    mol = Chem.RWMol(Chem.MolFromSmiles("O=c1ccc2c(o1)-c1ccccc1OC2"))
    new = [mol.AddAtom(Chem.Atom("C")) for _ in range(4)]
    path = [edge[0], *new, edge[1]]
    for index, (first, second) in enumerate(zip(path, path[1:])):
        mol.AddBond(first, second, Chem.BondType.DOUBLE if index in (1, 3) else Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    return mol


def _plan(rd_mol):
    mol = read_smiles(Chem.MolToSmiles(rd_mol, canonical=False))
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    return mol, atoms, result.plan


@pytest.mark.parametrize("change,count", (("extra_oxo", 0), ("spiro_at_8", 3), ("spiro_at_9", 3)))
@pytest.mark.parametrize("reverse", (False, True))
def test_added_hydrogen_count_is_derived_from_the_composed_graph(change, count, reverse):
    rd_mol = _spiro_variant(change)
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    result = name_mol(rd_mol, verify_opsin=opsin_available())
    assert result.error is None
    if opsin_available():
        assert result.opsin_check.status == "matched"
    if reverse:
        assert result.name == name_mol(_spiro_variant(change)).name
    _, _, plan = _plan(rd_mol)
    state = plan.derivative_state
    assert sum(len(operation.atom_ids) for operation in state.added_hydrogen_operations) == count
    assert not state.unsaturation_operations
    added = {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids}
    hydro = {atom for operation in state.hydro_operations for atom in operation.atom_ids}
    assert not added.intersection(hydro)


def test_multi_site_added_hydrogen_is_independently_bound():
    mol, atoms, plan = _plan(_spiro_variant("spiro_at_8"))
    delta = plan.derivative_state.bond_delta
    (operation,) = delta.added_hydrogen_operations
    assert len(operation.atom_ids) == 3
    corrupt = replace(operation, atom_ids=operation.atom_ids[:-1], locants=operation.locants[:-1])
    state = replace(plan.derivative_state, bond_delta=replace(delta, added_hydrogen_operations=(corrupt,)))
    audit = audit_fusion_plan(
        mol,
        atoms,
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        charge_operations=plan.charge_operations,
        derivative_state=state,
        mode=FusionMode.AUDITED_PIN,
    )
    assert audit.status is AuditStatus.MISMATCH
    assert "typed derivative state does not carry the selected parent bond delta" in audit.errors


@pytest.mark.parametrize("edge", ((8, 9), (9, 10), (10, 11)))
@pytest.mark.parametrize("reverse", (False, True))
def test_intrinsic_carbon_pair_survives_graph_built_benzene_annellation(edge, reverse):
    rd_mol = _annellated_carbon_pair(edge)
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    result = name_mol(rd_mol, verify_opsin=opsin_available())
    assert result.error is None
    if opsin_available():
        assert result.opsin_check.status == "matched"
    _, _, plan = _plan(rd_mol)
    assert len(plan.numbering.selected_face_model.faces) == 4
    (operation,) = plan.derivative_state.intrinsic_hydro_operations
    assert len(operation.atom_ids) == 2
    assert not plan.derivative_state.unsaturation_operations
    if reverse:
        assert result.name == name_mol(_annellated_carbon_pair(edge)).name
