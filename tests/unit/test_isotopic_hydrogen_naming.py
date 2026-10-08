import pytest

from openclatura import name


@pytest.mark.parametrize(
    "smiles,isotope_word",
    [
        ("[2H]C([2H])([2H])c1ccccc1", "trideuterio"),
        ("[3H]C([3H])([3H])c1ccccc1", "tritritio"),
    ],
)
def test_explicit_isotopic_hydrogens_are_named_recursively(smiles, isotope_word):
    result = name(smiles, verify_opsin=True)

    assert result.error is None
    assert isotope_word in result.name
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched"
