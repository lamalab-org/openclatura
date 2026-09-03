import pytest

from openclatura.fusion.model import SystemLocant
from openclatura.locants import canonical_locant_pair, parse_system_locant, system_locant_sort_key
from openclatura.molecule import Molecule


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4", SystemLocant(4)),
        ("4a", SystemLocant(4, "a")),
        ("9^2", SystemLocant(9, interior_distance=2)),
        ("9²", SystemLocant(9, interior_distance=2)),
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


def test_graph_mutation_invalidates_fusion_plans():
    mol = Molecule()
    mol.add_atom("C", idx=0)
    mol._fusion_plan_cache[("proof",)] = object()

    mol.add_atom("N", idx=1)

    assert not mol._fusion_plan_cache
