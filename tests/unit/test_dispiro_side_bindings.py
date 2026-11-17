"""Traced dispiro names keep the bindings of their terminal components."""

import pytest

from openclatura import name

CASES = [
    (
        "C1C2C1C1(CC1)C21CC1",
        "dispiro[cyclopropane-1,2'-bicyclo[2.1.0]pentane-3',1''-cyclopropane]",
    ),
    (
        "C1CC2(CCC13c1ccccc1-c1ccccc13)c1ccccc1-c1ccccc12",
        "dispiro[fluorene-9,1'-cyclohexane-4',9''-fluorene]",
    ),
]


@pytest.mark.parametrize(("smiles", "expected"), CASES)
def test_traced_dispiro_name_passes_the_final_audit(smiles, expected):
    plain = name(smiles)
    traced = name(smiles, include_trace=True)

    assert plain.name == expected
    assert traced.error is None
    assert traced.name == expected


@pytest.mark.parametrize(("smiles", "expected"), CASES)
def test_traced_dispiro_name_binds_both_side_components(smiles, expected):
    traced = name(smiles, include_trace=True)

    assembled = next(step for step in traced.decisions if step.decision == "assembled component name")
    bindings = assembled.data["name_atom_bindings"]
    sides = [binding for binding in bindings if binding["role"] == "spiro_substituent"]
    assert len(sides) == 2
    # Both sides are cited from the primed central component.
    assert all(locant.endswith("'") for binding in sides for locant in binding["locants"])
    central = next(binding for binding in bindings if binding["role"] == "parent")
    assert not set(central["atoms"]) & {atom for binding in sides for atom in binding["atoms"]}
