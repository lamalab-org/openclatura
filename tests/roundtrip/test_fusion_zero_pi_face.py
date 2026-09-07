"""A saturated fusion face can share aromatic edges with adjacent rings."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.graph_io import read_rdkit_mol
from openclatura.hantzsch_widman import hw_fusion_components_for_ring


def _dibenzocycle(symbol, hydroxy=False):
    mol = Chem.RWMol()
    for element in (symbol, "C", "C", "C", "C", "C", "C"):
        mol.AddAtom(Chem.Atom(element))
    for left in range(7):
        mol.AddBond(left, (left + 1) % 7, Chem.BondType.SINGLE)
    for left, right in ((1, 2), (3, 4)):
        path = [left, *(mol.AddAtom(Chem.Atom("C")) for _ in range(4)), right]
        for index, (a, b) in enumerate(zip(path, path[1:])):
            mol.AddBond(a, b, Chem.BondType.DOUBLE if index % 2 == 0 else Chem.BondType.SINGLE)
    if hydroxy:
        oxygen = mol.AddAtom(Chem.Atom("O"))
        mol.AddBond(8, oxygen, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("symbol", ("O", "S"))
def test_zero_observed_pi_bonds_do_not_remove_mancude_component(symbol):
    mol = Chem.RWMol()
    for element in (symbol, "C", "C", "C", "C", "C", "C"):
        mol.AddAtom(Chem.Atom(element))
    for left in range(7):
        mol.AddBond(left, (left + 1) % 7, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    graph = read_rdkit_mol(mol)
    matches = hw_fusion_components_for_ring(graph, list(range(7)))
    assert matches
    assert all(bond.bond_class == "mancude" for match in matches for bond in match.template.bonds)


@pytest.mark.opsin
@pytest.mark.parametrize("symbol", ("O", "S"))
@pytest.mark.parametrize("hydroxy", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_dibenzocycle_hydrogenation_and_side_only_descriptor_roundtrip(symbol, hydroxy, reverse):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    mol = _dibenzocycle(symbol, hydroxy)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    result = name_mol(mol, include_trace=True)
    assert result.error is None
    assert result.parent_nomenclature == "systematic_fusion"
    assert "6,7-dihydrodibenzo[b,d]" in result.name
    check = verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
