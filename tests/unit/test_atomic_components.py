import pytest

from openclatura import name, name_smiles
from openclatura.rules import elements


def test_graph_registry_carries_the_complete_periodic_table():
    assert len(elements.ELEMENTS) == 118
    assert elements.get("Ir").name == "iridium"
    assert not elements.get("Ir").nomenclature_supported
    assert elements.get("N").nomenclature_supported


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("[Ir]", "iridium(0)"),
        ("[Fe+2]", "iron(2+)"),
        ("[Pt+4]", "platinum(4+)"),
        ("[Al+3]", "aluminium(3+)"),
        ("[H-]", "hydride"),
    ],
)
def test_isolated_atomic_components_use_generic_charge_aware_names(smiles, expected):
    assert name_smiles(smiles) == expected


def test_existing_simple_ion_names_are_preserved():
    assert name_smiles("[Na+].[Cl-]") == "sodium chloride"
    assert name_smiles("[Mg+2].[Cl-].[Cl-]") == "magnesium dichloride"


def test_neutral_atomic_hydrogen_uses_element_name():
    assert name_smiles("[H]") == "hydrogen"


def test_isotopic_atomic_component_fails_instead_of_dropping_mass_number():
    result = name("[90Y]")
    assert result.name == ""
    assert "Isotopic atomic component naming is not supported: 90Y" in result.error


def test_bonded_unsupported_element_fails_before_organic_parent_selection():
    result = name("Cl[Zr+2]Cl")
    assert result.name == ""
    assert result.error.endswith("Bonded-element nomenclature is not supported for: Zr")


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("[H][H]", "dihydrogen"),
        ("N#N", "dinitrogen"),
        ("O=O", "dioxygen"),
        ("II", "diiodine"),
        ("[B-](F)(F)(F)F", "tetrafluoroborate"),
        ("[P-](F)(F)(F)(F)(F)F", "hexafluorophosphate"),
        ("O=S=O", "sulfur dioxide"),
        ("O=S(=O)=O", "sulfur trioxide"),
    ],
)
def test_complete_small_inorganic_graphs_use_retained_names(smiles, expected):
    result = name(smiles, verify_opsin=True)

    assert result.error is None
    assert result.name == expected
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched"


def test_implicit_hydrogen_dimer_uses_retained_name():
    assert name_smiles("[HH]") == "dihydrogen"


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CCO", "ethanol"),
        ("N#CON", "aminoxyformonitrile"),
    ],
)
def test_derived_organic_names_do_not_use_retained_component_graph_shortcut(smiles, expected):
    result = name(smiles, include_trace=True)

    assert result.name == expected
    assert all(step.decision != "matched retained component graph" for step in result.decisions)
