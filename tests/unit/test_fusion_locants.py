import pytest

from openclatura.fusion.model import SystemLocant
from openclatura.locants import (
    canonical_locant_pair,
    parse_system_locant,
    retained_locant_sort_key,
    system_locant_sort_key,
)
from openclatura.molecule import Molecule


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4", SystemLocant(4)),
        ("4a", SystemLocant(4, "a")),
        ("9^2", SystemLocant(9, interior_distance=2)),
        ("9²", SystemLocant(9, interior_distance=2)),
        ("3a^2", SystemLocant(3, fusion_suffix="a", interior_distance=2)),
        ("3a²", SystemLocant(3, fusion_suffix="a", interior_distance=2)),
    ],
)
def test_completed_system_locants_are_typed(text, expected):
    assert parse_system_locant(text) == expected


def test_completed_system_locants_reject_component_primes():
    with pytest.raises(ValueError, match="component prime"):
        parse_system_locant("2'")


def test_completed_system_locant_order_is_numeric_then_fusion_suffix():
    values = ["10", "4b", "4", "4a", "9^2"]
    assert sorted(values, key=system_locant_sort_key) == ["4", "4a", "4b", "9^2", "10"]
    assert canonical_locant_pair("4a", "4") == ("4", "4a")


def test_interior_distance_can_extend_a_fusion_anchor_locant():
    locant = SystemLocant(3, fusion_suffix="a", interior_distance=2)

    assert str(locant) == "3a²"
    assert locant.render(unicode_superscript=False) == "3a^2"


def test_graph_mutation_invalidates_fusion_plans():
    mol = Molecule()
    mol.add_atom("C", idx=0)
    mol._fusion_plan_cache[("proof",)] = object()

    mol.add_atom("N", idx=1)

    assert not mol._fusion_plan_cache


@pytest.mark.parametrize("text", ["5¹", "5^1", "5¹⁰", "5^10", "5a²", "5a^2", "4a", "10"])
def test_retained_sorter_uses_completed_system_locant_order(text):
    assert retained_locant_sort_key(text) == system_locant_sort_key(text)


def test_retained_sorter_orders_interior_distances_numerically():
    values = ["6", "5a²", "5¹⁰", "5a", "5²", "5", "5¹"]
    assert sorted(values, key=retained_locant_sort_key) == ["5", "5¹", "5²", "5¹⁰", "5a", "5a²", "6"]
    assert retained_locant_sort_key("5¹") == retained_locant_sort_key("5^1")


def test_retained_sorter_preserves_legacy_label_support():
    values = ["N", "10", "4a'", "4a", "4'", "4", "O"]
    assert sorted(values, key=retained_locant_sort_key) == ["4", "4'", "4a", "4a'", "10", "N", "O"]
