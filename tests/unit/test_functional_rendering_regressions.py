"""Functional rendering regressions from the PubChem 1/2 report."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.component_modifiers import n_substituent_locant
from openclatura.formatting import format_multiplier

REPORTED_CASES = (
    (14327, "CCCCNC(=NC)/C(=C/N)C(N)=Nc1ccc(C)cc1C"),
    (17325, "CCCCOC(=O)/C=C\\C(=O)OCCCCCCCCCCOC(=O)/C=C\\C(=O)OCCCC"),
    (24970, "O=C(OC/C=C/C(F)(F)F)N(S)S"),
    (26788, "COC1CCCCC1/C=N/N=C(\\N)N[N+](=O)[O-]"),
    (37125, "CCCC(=O)N(C=O)CCCOC(=O)/C=C/C(=O)OC"),
    (52013, "CCOc1ccc(S(=O)(=O)N/N=C\\c2c(F)c(F)c(F)c(F)c2F)cc1"),
    (52695, "CCOC(=O)N(S)S"),
    (73522, "COc1cc2ncnc(N=S(C)(=O)c3ccc(NC(N)=O)cc3)c2cc1OC"),
    (77615, "CC(C)COCOP(=O)(O)OCOCC(C)C"),
    (87026, "CC(C)(C)COc1ccc(-c2ccc3c(C#N)nc(C(P)(P)P)n3c2)cn1"),
    (87450, "CC(C)(C)[S@](=O)NC(c1cc(/C(N)=N/O)cs1)C1CC1"),
)


def _assert_exact(mol):
    result = name_mol(mol, include_trace=True)
    assert result.error is None, result.error
    check = verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False)
    assert check.status == "matched", (result.name, check.to_dict())
    assert check.canonical_original == check.canonical_roundtrip
    return result


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java required")
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("index,smiles", REPORTED_CASES, ids=[str(index) for index, _ in REPORTED_CASES])
def test_reported_functional_rendering_is_graph_exact(index, smiles, reverse):
    mol = Chem.MolFromSmiles(smiles)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    _assert_exact(mol)


VARIANT_CORES = (
    "COC(=O)N(S)S",
    "COC(=O)C=CC(=O)OCCOC",
    "COCO[P](=O)(O)OCOC",
    "CN=C(NC)CC(N)=NCC",
    "CS(=O)(=O)N/N=C/c1ccccc1",
    "CS(=O)(=O)N/N=C\\c1ccccc1",
    "C[S@](=O)NC(C)CC(=O)O",
    "CC(C)N=S(C)(=O)c1ccc(N)cc1",
    "OCCNN=NN",
    "COC1CCCCC1/C=N/N=C(\\N)NC",
)


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java required")
@pytest.mark.parametrize("extend", (False, True))
@pytest.mark.parametrize("smiles", VARIANT_CORES)
def test_graph_built_functional_variants_roundtrip(smiles, extend):
    mol = Chem.MolFromSmiles(smiles)
    if extend:
        terminal = max((a for a in mol.GetAtoms() if a.GetSymbol() == "C"), key=lambda a: a.GetTotalNumHs()).GetIdx()
        builder = Chem.RWMol(mol)
        carbon = builder.AddAtom(Chem.Atom("C"))
        builder.AddBond(terminal, carbon, Chem.BondType.SINGLE)
        mol = builder.GetMol()
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    _assert_exact(mol)


@pytest.mark.parametrize(
    "prefix,count,expected",
    (
        ("sulfanyl", 1, "sulfanyl"),
        ("sulfanyl", 2, "bis(sulfanyl)"),
        ("sulfanyl", 3, "tris(sulfanyl)"),
        ("phosphanyl", 3, "tris(phosphanyl)"),
        ("isobutoxymethyl", 2, "bis(isobutoxymethyl)"),
        ("cyclopropylmethyl", 2, "bis(cyclopropylmethyl)"),
        ("methyl", 2, "dimethyl"),
        ("ethoxy", 2, "diethoxy"),
        ("amino", 2, "diamino"),
        ("3-methoxypropyl", 1, "(3-methoxypropyl)"),
    ),
)
def test_multiplier_preserves_ligand_boundaries(prefix, count, expected):
    assert format_multiplier(prefix, count) == expected


def test_diamidine_primes_distinguish_amino_and_imino_sites():
    assert [
        n_substituent_locant("amidine", 2, 2, nitrogen, group * 2 + nitrogen, group)
        for group in range(2)
        for nitrogen in range(2)
    ] == ["N", "N''", "N'", "N'''"]
