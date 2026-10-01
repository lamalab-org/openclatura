"""Retained parents outrank assembled components, and symmetric ones number stably.

Three defects met where a retained fused parent is chosen. A spiro side took a
fusion hydride without asking whether a retained parent outranked it; a parent
whose feature sat on a ring-fusion locant was declined outright; and a parent
with a symmetry, such as fluorene's two benzo rings, left its numbering to the
order the input atoms happened to arrive in.
"""

import pytest
from rdkit import Chem

from openclatura import name_mol, name_smiles, opsin_available, verify_with_opsin

# Symmetric retained parents: either benzo ring can carry either substituent.
SYMMETRIC = (
    ("COc1ccc2c(c1)Cc1cc(F)ccc12", "2-fluoro-7-methoxy-9H-fluorene"),
    ("COc1ccc2c(c1)Cc1cc(F)ccc1O2", "2-fluoro-7-methoxy-9H-xanthene"),
    ("COc1ccc2c(c1)[nH]c1cc(F)ccc12", "2-fluoro-7-methoxy-9H-carbazole"),
    ("COc1ccc2nc3ccc(F)cc3cc2c1", "2-fluoro-7-methoxyacridine"),
    ("COc1ccc2c(c1)oc1cc(F)ccc12", "3-fluoro-7-methoxydibenzofuran"),
    ("COc1ccc2cc3cc(F)ccc3cc2c1", "2-fluoro-6-methoxyanthracene"),
)

# A retained parent reached through a spiro junction, or carrying a feature on a
# ring-fusion locant, which the constructed reading would spell from components.
RETAINED_OVER_CONSTRUCTED = (
    ("COC(=O)C[C@]1(C)CC[C@@]2(O1)C(COC(C)=O)=CC[C@H]1C(C)(C)CCC[C@@]12C", "naphthalene", "benzobenzene"),
    ("CCO[C@@H]1CCC2(C)C(CC[C@@]23CCC(=O)O3)C12C=CC1=CC(=O)CCC1(C)C2", "naphthalene", "benzobenzene"),
    ("Cc1ccc(F)c2c1C1(CN2)CC1N", "indole", "benzo[b]pyrrole"),
    ("CCCNC(=O)N1CCC2(CC1)Nc1ccccc1C(=O)N2Cc1ccccc1", "quinazoline", "benzo[d]pyrimidine"),
    ("COc1ccc2c(c1)C1(COC(N)=N1)c1cc(-c3cnccc3F)ccc1O2", "xanthene", "dibenzo[b,e]pyran"),
)


def _reversed(mol):
    return Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))


@pytest.mark.parametrize(("smiles", "expected"), SYMMETRIC)
def test_a_symmetric_retained_parent_numbers_the_same_either_way(smiles, expected):
    """P-14.3.5 settles the tie, so the input atom order cannot."""

    mol = Chem.MolFromSmiles(smiles)
    assert name_mol(mol).name == expected
    assert name_mol(_reversed(mol)).name == expected


@pytest.mark.parametrize(("smiles", "expected"), SYMMETRIC)
def test_the_symmetric_tie_gives_the_first_cited_prefix_the_low_locant(smiles, expected):
    """fluoro precedes methoxy alphanumerically, so it takes the lower locant."""

    name = name_smiles(smiles)
    assert name.index("fluoro") < name.index("methoxy")


def test_alphanumerical_order_ignores_enclosing_marks():
    """A bracketed prefix must not sort ahead of bromo on its bracket."""

    name = name_smiles("CCc1ccc(Cc2cc3c(cc2Br)CO[C@]32O[C@H](C)[C@@H](O)[C@H](O)[C@H]2O)s1")
    assert "5-bromo" in name, name


@pytest.mark.parametrize(("smiles", "retained", "constructed"), RETAINED_OVER_CONSTRUCTED)
def test_a_retained_parent_outranks_its_assembled_components(smiles, retained, constructed):
    name = name_smiles(smiles)
    assert retained in name, name
    assert constructed not in name


@pytest.mark.parametrize(("smiles", "retained", "constructed"), RETAINED_OVER_CONSTRUCTED)
def test_those_names_are_order_invariant(smiles, retained, constructed):
    mol = Chem.MolFromSmiles(smiles)
    assert name_mol(mol).name == name_mol(_reversed(mol)).name


@pytest.mark.skipif(not opsin_available(), reason="OPSIN round-trip needs java and py2opsin")
@pytest.mark.parametrize("smiles", [s for s, _ in SYMMETRIC] + [s for s, _, _ in RETAINED_OVER_CONSTRUCTED])
def test_every_such_name_round_trips(smiles):
    assert verify_with_opsin(name_smiles(smiles), smiles).status == "matched"
