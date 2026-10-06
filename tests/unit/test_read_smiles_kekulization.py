"""An aromatic ring without a Kekule structure is refused, not read as saturated."""

import pytest
from rdkit import Chem

from openclatura import name, name_mol
from openclatura.graph_io import read_rdkit_mol, read_smiles


def _double_bonds(mol) -> int:
    return sum(bond.order == 2 for bond in mol.bonds.values())


# Pyrrole and indole written without their [nH]: RDKit cannot kekulize them,
# and they used to be named pyrrolidine and octahydro-1H-indole
# (lamalab-org/openclatura#140).
UNKEKULIZABLE = ["c1ccnc1", "c1ccc2nccc2c1", "CCO.c1ccnc1"]


@pytest.mark.parametrize("smiles", UNKEKULIZABLE)
def test_read_smiles_refuses_an_aromatic_ring_without_a_kekule_structure(smiles):
    with pytest.raises(ValueError, match="no Kekule structure"):
        read_smiles(smiles)


@pytest.mark.parametrize("smiles", UNKEKULIZABLE)
def test_naming_an_unkekulizable_smiles_fails_cleanly(smiles):
    result = name(smiles)
    assert result.ok is False
    assert result.name == ""
    assert "no Kekule structure" in result.error


def test_name_mol_refuses_an_unsanitized_unkekulizable_molecule():
    result = name_mol(Chem.MolFromSmiles("c1ccnc1", sanitize=False))
    assert result.ok is False
    assert "no Kekule structure" in result.error


def test_read_rdkit_mol_still_kekulizes_an_unsanitized_aromatic_molecule():
    mol = read_rdkit_mol(Chem.MolFromSmiles("c1ccccc1", sanitize=False))
    assert _double_bonds(mol) == 3


def test_written_pyrrole_still_reads():
    assert name("c1cc[nH]c1").name == "1H-pyrrole"


def test_unsanitizable_valence_still_falls_back_to_the_graph_as_written():
    # RDKit rejects the five-coordinate carbon; the namer reads it anyway (#112).
    mol = read_smiles("CCCC(C)(C)(C)C")
    assert len(mol.atoms) == 8
    assert name("CCCC(C)(C)(C)C").ok is True


def test_valence_fallback_keeps_the_aromatic_ring():
    # The fallback used to skip kekulization, so the benzene ring next to the
    # five-coordinate carbon was read, and named, as cyclohexane.
    mol = read_smiles("c1ccccc1C(C)(C)(C)(C)C")
    assert _double_bonds(mol) == 3
    assert name("c1ccccc1C(C)(C)(C)(C)C").name == name("C1=CC=CC=C1C(C)(C)(C)(C)C").name
