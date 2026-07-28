"""Regression tests for systematic chain-length stems."""

import pytest

from openclatura import name
from openclatura.formatting import substituted_alkoxy_prefix
from openclatura.ring_renderer import von_baeyer_cycle_count
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


@pytest.mark.parametrize("length", range(1, 1001))
def test_terminal_stem_round_trip(length):
    stem = stems.get(length)
    assert stems.terminal_stem(f"2-hydroxy{stem.stem}yl") == stem


@pytest.mark.parametrize("length", range(1, 1001))
def test_basic_prefix_round_trip(length):
    assert stems.length_for_basic_prefix(f"{stems.stem_for(length)}a") == length


@pytest.mark.parametrize("value", [None, "", 31, "not-a-stem", "ethylated"])
def test_reverse_stem_lookups_reject_unknown_values(value):
    assert stems.terminal_stem(value) is None
    assert stems.length_for_basic_prefix(value) is None


@pytest.mark.parametrize(
    "branch, expected",
    [
        ("2-hydroxyethyl", "(2-hydroxyethoxy)"),
        ("2-hydroxyhentriacontyl", "(2-hydroxyhentriacontoxy)"),
        ("2-hydroxyhexaoctacontatetractyl", "(2-hydroxyhexaoctacontatetractoxy)"),
        ("2-hydroxykiliyl", "(2-hydroxykilioxy)"),
    ],
)
def test_substituted_alkoxy_prefix_uses_longest_generated_terminal_stem(branch, expected):
    assert substituted_alkoxy_prefix(branch) == expected


@pytest.mark.parametrize("branch", ["ethyl", "2-hydroxyalkyl", "hydroxyethylated"])
def test_substituted_alkoxy_prefix_rejects_unrelated_branches(branch):
    assert substituted_alkoxy_prefix(branch) is None


@pytest.mark.parametrize(
    "count",
    [3, 10, 31, 100, 486, 1000],
)
def test_von_baeyer_cycle_count_recognizes_generated_prefixes(count):
    prefix = f"{stems.stem_for(count)}a"
    assert von_baeyer_cycle_count(f"{prefix}cyclo[1.1.1]") == count


@pytest.mark.parametrize("descriptor", [None, "", "notacyclo[1.1.1]", "hentriacont[1.1.1]"])
def test_von_baeyer_cycle_count_rejects_unknown_prefixes(descriptor):
    assert von_baeyer_cycle_count(descriptor) is None


@pytest.mark.slow
@pytest.mark.parametrize("length", range(1, 1001))
def test_linear_alkanes_from_one_through_one_thousand(length):
    result = name("C" * length)
    expected = stems.stem_for(length) + "ane"
    assert result.error is None, f"C{length}: {result.error}"
    assert result.name == expected
