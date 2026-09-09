"""Component primes precede lambda bonding numbers in spiro replacement names."""

from copy import deepcopy

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.assembly_parts import AssemblyParts, SubstituentItem
from openclatura.assembly_spiro import _prime_side_locant, spiro_assembly_from_parts
from openclatura.opsin_verify import verify_with_opsin


@pytest.mark.parametrize(
    "locant,expected",
    (
        ("", ""),
        ("2", "2'"),
        ("3a", "3a'"),
        ("N", "N'"),
        ("2'", "2'"),
        ("2lambda^6", "2'lambda^6"),
        ("7lambda^4", "7'lambda^4"),
        ("3alambda^5", "3a'lambda^5"),
        ("2'lambda^6", "2'lambda^6"),
        ("2''lambda^6", "2''lambda^6"),
    ),
)
def test_component_prime_precedes_annotation_and_is_idempotent(locant, expected):
    assert _prime_side_locant(locant) == expected
    assert _prime_side_locant(expected) == expected


def test_lambda_projection_uses_component_metadata_without_mutating_it(monkeypatch):
    import openclatura.assembly_spiro as assembly

    def reject_parse(*args, **kwargs):
        pytest.fail("component replacements must not be recovered from rendered names")

    monkeypatch.setattr(assembly, "_split_side_prefix_run", reject_parse)
    local = AssemblyParts(
        parent_length=4,
        is_ring=True,
        a_prefixes=[SubstituentItem("thia", ["2lambda^6"], atom_ids={1})],
        substituents=[SubstituentItem("oxo", ["2", "2"], atom_ids={4, 5}, bond_ids={4, 5})],
    )
    original = deepcopy(local)
    side = spiro_assembly_from_parts(local, "4")
    assert side.side_prefixes == ("2',2'-oxo", "2'lambda^6-thia")
    assert side.side_substituents[0].atom_ids == {4, 5}
    assert side.side_substituents[0].bond_ids == {4, 5}
    assert side.side_parts == original
    assert local == original


CASES = (
    pytest.param(
        "[C-]#[N+]c1cccc(-c2ccc3c(c2)C2(CC(c4ccccc4)S3(=O)=O)N=C(N)N(C)C2=O)c1",
        "2'lambda^6-thia",
        id="pubchem-2579-sulfone",
    ),
    pytest.param(
        "CC1=C(C)C2(CN(C(C)(C)C)S2(C)CCCCCOC(C)(C)C)c2c1sc1ccccc21",
        "1'lambda^4-thia",
        id="pubchem-23944-nonoxo-sulfur",
    ),
    pytest.param(
        "O=C1CC2S(=O)C1C(c1cccc(Cl)c1)[C@]21C(=O)Nc2cc(Cl)ccc21",
        "7'lambda^4-thia",
        id="pubchem-54830-sulfoxide-stereo",
    ),
    pytest.param(
        "CC12CCC(C3(OS(=O)(=O)O3)C1=O)C2(C)C",
        "2'lambda^6-thia",
        id="pubchem-85807-sulfate",
    ),
)


@pytest.mark.parametrize("smiles,annotation", CASES)
def test_public_lambda_names_are_atom_order_invariant(smiles, annotation):
    mol = Chem.MolFromSmiles(smiles)
    forward = name_mol(mol)
    reverse = name_mol(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))))
    assert forward.error is None
    assert reverse.error is None
    assert annotation in forward.name
    assert reverse.name == forward.name


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles,annotation", CASES)
def test_public_lambda_names_roundtrip_without_standardization(smiles, annotation):
    result = name_mol(Chem.MolFromSmiles(smiles))
    assert result.error is None
    assert annotation in result.name
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    assert check.status == "matched", check
    assert check.canonical_original == check.canonical_roundtrip
