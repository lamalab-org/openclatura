"""Regression tests for systematic chain-length stems."""

import pytest

from openclatura import name
from openclatura.rules import stems

LEGACY_STEMS = (
    "meth",
    "eth",
    "prop",
    "but",
    "pent",
    "hex",
    "hept",
    "oct",
    "non",
    "dec",
    "undec",
    "dodec",
    "tridec",
    "tetradec",
    "pentadec",
    "hexadec",
    "heptadec",
    "octadec",
    "nonadec",
    "icos",
    "henicos",
    "docos",
    "tricos",
    "tetracos",
    "pentacos",
    "hexacos",
    "heptacos",
    "octacos",
    "nonacos",
    "triacont",
)


@pytest.mark.parametrize("length, expected", enumerate(LEGACY_STEMS, start=1))
def test_legacy_stems_are_unchanged(length, expected):
    assert stems.stem_for(length) == expected


@pytest.mark.parametrize(
    "length, expected",
    [
        (31, "hentriacont"),
        (39, "nonatriacont"),
        (40, "tetracont"),
        (41, "hentetracont"),
        (52, "dopentacont"),
        (99, "nonanonacont"),
        (100, "hect"),
        (101, "henhect"),
        (111, "undecahect"),
        (199, "nonanonacontahect"),
        (200, "dict"),
        (363, "trihexacontatrict"),
        (486, "hexaoctacontatetract"),
        (999, "nonanonacontanonact"),
        (1000, "kili"),
    ],
)
def test_systematic_stem_examples(length, expected):
    assert stems.stem_for(length) == expected
    assert stems.get(length) == stems.Stem(length, expected, retained=False)


@pytest.mark.parametrize("length", [None, 1.0, "31", [], {}, True, False, 0, -1, 1001])
def test_invalid_stem_lengths_raise_value_error(length):
    with pytest.raises(ValueError, match="1 through 1000"):
        stems.stem_for(length)


def test_generated_stems_are_unique():
    generated = [stems.stem_for(length) for length in range(1, 1001)]
    assert len(generated) == len(set(generated)) == 1000


@pytest.mark.slow
@pytest.mark.parametrize("length", range(1, 1001))
def test_linear_alkanes_from_one_through_one_thousand(length):
    result = name("C" * length)
    expected = stems.stem_for(length) + "ane"
    assert result.error is None, f"C{length}: {result.error}"
    assert result.name == expected
