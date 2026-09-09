"""A saturated N-H must not suppress a separate aromatic lone-pair donor."""

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol, opsin_available
from openclatura.fusion.indicated_hydrogen import aromatic_nitrogen_hydrogen_atoms
from openclatura.fusion.mancude import indicated_hydrogen_parent_bond_model, parent_derivative_state
from openclatura.fusion.model import FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.graph_io import read_smiles
from openclatura.opsin_verify import verify_with_opsin

# These expected names were checked with OPSIN without standardization.
CASES = (
    ("C1CC2=C(N1)C=CN2", "2,3-dihydro-1H,4H-pyrrolo[3,2-b]pyrrole"),
    ("c1cc2c([nH]1)CCN2", "2,3-dihydro-1H,4H-pyrrolo[3,2-b]pyrrole"),
    ("C1CC2=CNC=C2N1", "2,3-dihydro-1H,5H-pyrrolo[3,4-b]pyrrole"),
    ("c1[nH]cc2c1CCN2", "2,3-dihydro-1H,5H-pyrrolo[3,4-b]pyrrole"),
)
SMILES = tuple(smiles for smiles, _ in CASES)


def _case(smiles=SMILES[0]):
    mol = read_smiles(smiles)
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom.idx, atom.symbol) for atom in mol.atoms.values()),
        bonds=tuple(FusionGraphBond((bond.u, bond.v)) for bond in mol.bonds.values()),
    )
    return mol, graph


@pytest.mark.parametrize("smiles", SMILES)
def test_mixed_nh_keeps_aromatic_donor_and_saturated_hydro_pair(smiles):
    mol, graph = _case(smiles)
    aromatic_nh = frozenset(atom.idx for atom in mol.atoms.values() if atom.symbol == "N" and atom.is_aromatic)
    nitrogen = frozenset(atom.idx for atom in mol.atoms.values() if atom.symbol == "N")
    assert len(aromatic_nh) == 1 and len(nitrogen) == 2
    assert aromatic_nitrogen_hydrogen_atoms(mol, graph) == aromatic_nh
    model = indicated_hydrogen_parent_bond_model(graph, aromatic_nh)
    assert model.maximum_non_cumulative_double_bonds == 3
    assert all(
        order == 1
        for assignment in model.allowed_kekule_assignments
        for edge, order in assignment.orders
        if aromatic_nh.intersection(edge)
    )
    state = parent_derivative_state(
        mol,
        frozenset(mol.atoms),
        model,
        {atom: str(atom + 1) for atom in mol.atoms},
        indicated_hydrogen_atom_ids=nitrogen,
    )
    assert state is not None
    assert len(state.bond_delta.hydrogenated_edges) == 1
    assert all(
        mol.atoms[atom].symbol == "C" and not mol.atoms[atom].is_aromatic
        for atom in state.bond_delta.hydrogenated_atom_ids
    )
    assert all(order == 1 for edge, order in state.bond_delta.assignment.orders if nitrogen.intersection(edge))
    assert not state.unsaturation_operations
    assert not state.oxo_operations


@pytest.mark.parametrize("change", ("charged", "no_h", "non_aromatic", "double_bond", "external_oxo"))
def test_mixed_nh_donor_still_requires_local_valence_and_external_scope(change):
    mol, graph = _case()
    donor = next(atom.idx for atom in mol.atoms.values() if atom.symbol == "N" and atom.is_aromatic)
    if change == "charged":
        mol.update_atom(donor, charge=1)
    elif change == "no_h":
        mol.update_atom(donor, total_h_count=0)
    elif change == "non_aromatic":
        mol.update_atom(donor, is_aromatic=False)
    elif change == "double_bond":
        mol.update_bond(mol.get_bond(donor, mol.get_neighbors(donor)[0]).idx, order=2)
    else:
        carbon = next(atom.idx for atom in mol.atoms.values() if atom.symbol == "C" and not atom.is_aromatic)
        oxygen = max(mol.atoms) + 1
        mol.add_atom("O", idx=oxygen)
        mol.add_bond(carbon, oxygen, idx=max(mol.bonds) + 1, order=2)
    assert not aromatic_nitrogen_hydrogen_atoms(mol, graph)


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize("smiles,expected", CASES)
def test_mixed_nh_exact_opsin_name_survives_atom_reversal(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    canonical_input = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
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
        assert check.canonical_original == check.canonical_roundtrip == canonical_input
        decoded = Chem.MolFromSmiles(check.opsin_smiles)
        assert decoded is not None
        assert Chem.MolToSmiles(decoded, canonical=True, isomericSmiles=True) == canonical_input
