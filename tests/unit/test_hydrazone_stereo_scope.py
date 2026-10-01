"""Sulfonyl substitution must not discard the hydrazone's E/Z descriptor."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.graph_io import read_smiles
from openclatura.perception import PerceivedGroup
from openclatura.principal_groups import _hydrazone_allows_unlocanted_stereo


@pytest.mark.parametrize(
    "smiles,expected",
    [
        (
            r"CCOc1ccc(S(=O)(=O)N/N=C\c2c(F)c(F)c(F)c(F)c2F)cc1",
            "(Z)-N-(4-ethoxyphenylsulfonyl)-2,3,4,5,6-pentafluorobenzaldehyde hydrazone",
        ),
        (r"CS(=O)(=O)N/N=C\c1ccccc1", "(Z)-N-(methylsulfonyl)benzaldehyde hydrazone"),
        (r"CS(=O)(=O)N/N=C/c1ccccc1", "(E)-N-(methylsulfonyl)benzaldehyde hydrazone"),
        (r"CS(=O)(=O)N/N=C\C", "(Z)-N-(methylsulfonyl)acetaldehyde hydrazone"),
        ("CS(=O)(=O)NN=Cc1ccccc1", "N-(methylsulfonyl)benzaldehyde hydrazone"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_sulfonyl_hydrazone_stereo_is_generated_and_graph_exact(smiles, expected, reverse):
    mol = Chem.MolFromSmiles(smiles)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    result = name_mol(mol)
    assert result.error is None
    assert result.name == expected
    if opsin_available():
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip


@pytest.mark.parametrize(
    "corruption",
    ["sulfur_charge", "nitrogen_charge", "oxygen_charge", "sulfinyl", "sulfur_imide", "oxygen_bridge", "hetero_ligand"],
)
def test_sulfonyl_stereo_scope_does_not_admit_other_sulfur_states(corruption):
    mol = read_smiles(r"CS(=O)(=O)N/N=C\C")
    group = PerceivedGroup("aldehyde_hydrazone", True, 6, {4, 5, 6})
    assert _hydrazone_allows_unlocanted_stereo(mol, group, 6)
    if corruption == "sulfur_charge":
        mol.update_atom(1, charge=1)
    elif corruption == "nitrogen_charge":
        mol.update_atom(4, charge=1)
    elif corruption == "oxygen_charge":
        mol.update_atom(2, charge=-1)
    elif corruption == "sulfinyl":
        mol.update_bond(mol.get_bond(1, 2).idx, order=1)
    elif corruption == "sulfur_imide":
        mol.update_bond(mol.get_bond(1, 4).idx, order=2)
    elif corruption == "oxygen_bridge":
        mol.add_bond(2, 0)
    else:
        mol.update_atom(0, symbol="N")
    assert not _hydrazone_allows_unlocanted_stereo(mol, group, 6)
