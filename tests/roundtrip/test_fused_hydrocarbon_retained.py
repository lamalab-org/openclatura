"""Graph-backed retained-name coverage for the requested fused hydrocarbons."""

from __future__ import annotations

import shutil
import tempfile
import warnings
import xml.etree.ElementTree as ET

import pytest
from rdkit import Chem

from openclatura import name_many, name_smiles
from openclatura.retained_fused_templates import (
    _acene_template_from_data,
    _generated_acene_templates,
    _generated_polyaphene_templates,
    retained_fused_graph_templates,
)
from openclatura.retained_name_policy import retained_parent_name_policy
from openclatura.rules import multipliers
from openclatura.utils import standardize_mol

try:
    import py2opsin
except Exception:  # pragma: no cover - optional test dependency
    py2opsin = None


FUSED_HYDROCARBON_CASES = (
    ("naphthalene", "naphthalene", "c1ccc2ccccc2c1"),
    ("acenaphthylene", "acenaphthylene", "C1=Cc2cccc3cccc1c23"),
    ("acenaphthene", "1,2-dihydroacenaphthylene", "c1cc2c3c(cccc3c1)CC2"),
    ("fluorene", "9H-fluorene", "c1ccc2c(c1)Cc1ccccc1-2"),
    ("phenanthrene", "phenanthrene", "c1ccc2c(c1)ccc1ccccc12"),
    ("anthracene", "anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("fluoranthene", "fluoranthene", "c1ccc2c(c1)-c1cccc3cccc-2c13"),
    ("pyrene", "pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
    ("benz[a]anthracene", "benz[a]anthracene", "c1ccc2cc3c(ccc4ccccc43)cc2c1"),
    ("chrysene", "chrysene", "c1ccc2c(c1)ccc1c3ccccc3ccc21"),
    ("tetracene", "tetracene", "c1ccc2cc3cc4ccccc4cc3cc2c1"),
    ("pentacene", "pentacene", "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1"),
    ("hexacene", "hexacene", "c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1"),
    ("pentaphene", "pentaphene", "c1ccc2cc3c(ccc4cc5ccccc5cc43)cc2c1"),
    ("hexaphene", "hexaphene", "c1ccc2cc3cc4c(ccc5cc6ccccc6cc54)cc3cc2c1"),
    ("heptaphene", "heptaphene", "c1ccc2cc3cc4c(ccc5cc6cc7ccccc7cc6cc54)cc3cc2c1"),
    ("octaphene", "octaphene", "c1ccc2cc3cc4cc5c(ccc6cc7cc8ccccc8cc7cc65)cc4cc3cc2c1"),
    ("tetraphenylene", "tetraphenylene", "c1ccc2c(c1)-c1ccccc1-c1ccccc1-c1ccccc1-2"),
    ("rubicene", "rubicene", "c1ccc2c(c1)c1cccc3c4c5ccccc5c5cccc(c2c13)c54"),
    ("trinaphthylene", "trinaphthylene", "c1ccc2cc3c(cc2c1)c1cc2ccccc2cc1c1cc2ccccc2cc31"),
    ("pyranthrene", "pyranthrene", "c1ccc2c(c1)cc1ccc3cc4c5ccccc5cc5ccc6cc2c1c3c6c54"),
    ("ovalene", "ovalene", "c1cc2ccc3cc4ccc5ccc6ccc7cc8ccc1c1c2c3c2c4c5c6c7c2c81"),
    ("benzo[e]pyrene", "benzo[e]pyrene", "c1ccc2c(c1)c1cccc3ccc4cccc2c4c31"),
    ("benzo[j]fluoranthene", "benzo[j]fluoranthene", "c1ccc2c3c(ccc2c1)-c1cccc2cccc-3c12"),
    ("cyclopenta[cd]pyrene", "cyclopenta[cd]pyrene", "C1=Cc2cc3cccc4ccc5ccc1c2c5c43"),
    ("benzo[b]fluoranthene", "benzo[b]fluoranthene", "c1ccc2c(c1)-c1cccc3c1c-2cc1ccccc13"),
    ("benzo[k]fluoranthene", "benzo[k]fluoranthene", "c1ccc2cc3c(cc2c1)-c1cccc2cccc-3c12"),
    ("benzo[a]pyrene", "benzo[a]pyrene", "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"),
    (
        "indeno[1,2,3-cd]pyrene",
        "indeno[1,2,3-cd]pyrene",
        "c1ccc2c(c1)-c1ccc3ccc4cccc5cc-2c1c3c45",
    ),
    (
        "dibenz[a,h]anthracene",
        "dibenz[a,h]anthracene",
        "c1ccc2c(c1)ccc1cc3c(ccc4ccccc43)cc12",
    ),
    ("benzo[ghi]perylene", "benzo[ghi]perylene", "c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56"),
)

NEW_TEMPLATE_NAMES = (
    "benz[a]anthracene",
    "benzo[a]pyrene",
    "benzo[b]fluoranthene",
    "benzo[k]fluoranthene",
    "indeno[1,2,3-cd]pyrene",
    "benzo[ghi]perylene",
    "pentaphene",
    "hexaphene",
    "heptaphene",
    "octaphene",
    "tetraphenylene",
    "rubicene",
    "trinaphthylene",
    "pyranthrene",
    "ovalene",
    "benzo[e]pyrene",
    "benzo[j]fluoranthene",
    "cyclopenta[cd]pyrene",
)


@pytest.mark.parametrize(("source_name", "expected_name", "smiles"), FUSED_HYDROCARBON_CASES)
def test_requested_fused_hydrocarbons_use_retained_parent(source_name, expected_name, smiles):
    assert name_smiles(smiles) == expected_name, source_name


@pytest.mark.parametrize(("source_name", "expected_name", "smiles"), FUSED_HYDROCARBON_CASES)
def test_requested_fused_parent_name_is_invariant_to_atom_order(source_name, expected_name, smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, source_name
    for _ in range(3):
        reordered = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        assert name_smiles(reordered) == expected_name, (source_name, reordered)


def test_higher_acene_series_generates_standard_locant_graphs():
    generated = {template.name: template for template in _generated_acene_templates()}
    expected_sizes = {
        "tetracene": (18, 21, 4),
        "pentacene": (22, 26, 5),
        "hexacene": (26, 31, 6),
        "heptacene": (30, 36, 7),
        "octacene": (34, 41, 8),
        "nonacene": (38, 46, 9),
    }
    assert {
        name: (len(generated[name].atoms), len(generated[name].bonds), len(generated[name].rings))
        for name in expected_sizes
    } == expected_sizes

    registered = {template.name: template for template in retained_fused_graph_templates()}
    assert registered["tetracene"].aliases == ("naphthacene",)
    assert registered["hexacene"].numbering_policy == "generated_acene_series"
    assert {frozenset(bond.locants) for bond in generated["pentacene"].bonds} == {
        frozenset(bond.locants) for bond in registered["pentacene"].bonds
    }


@pytest.mark.parametrize(
    ("expected_name", "smiles"),
    (
        ("heptacene", "c1ccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc2c1"),
        ("octacene", "c1ccc2cc3cc4cc5cc6cc7cc8ccccc8cc7cc6cc5cc4cc3cc2c1"),
        ("nonacene", "c1ccc2cc3cc4cc5cc6cc7cc8cc9ccccc9cc8cc7cc6cc5cc4cc3cc2c1"),
    ),
)
def test_higher_acene_series_extends_without_new_graph_templates(expected_name, smiles):
    assert name_smiles(smiles) == expected_name


def test_higher_acene_series_is_synthesized_on_demand_beyond_eager_registry():
    smiles = (
        "C1C=CC2C=C3C=C4C=C5C=C6C=C7C=C8C=C9C=C%10C=C%11C=C%12C=C%13C=CC=CC%13=CC%12="
        "CC%11=CC%10=CC9=CC8=CC7=CC6=CC5=CC4=CC3=CC=2C=1"
    )
    assert name_smiles(smiles) == "tridecacene"

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    for _ in range(2):
        reordered = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        assert name_smiles(reordered) == "tridecacene"


def test_generated_acene_name_uses_shared_numerical_terms_beyond_explicit_multiplier_table():
    template = _acene_template_from_data({"ring_count": 21, "name": f"{multipliers.basic(21)}cene"})
    assert template.name == "henicosacene"
    assert len(template.atoms) == 86
    assert len(template.rings) == 21


def test_higher_polyaphene_series_is_synthesized_on_demand_beyond_eager_registry():
    smiles = (
        "C1=CC=CC2=CC3=CC4=CC5=CC6=CC7=CC=C8C=C9C=C%10C=C%11C=C%12C=C%13C=CC=CC%13=CC%12="
        "CC%11=CC%10=CC9=CC8=C7C=C6C=C5C=C4C=C3C=C12"
    )
    assert name_smiles(smiles) == "tridecaphene"


def test_polyaphene_series_generates_standard_locant_graphs():
    generated = {template.name: template for template in _generated_polyaphene_templates()}
    expected_sizes = {
        "pentaphene": (22, 26, 5),
        "hexaphene": (26, 31, 6),
        "heptaphene": (30, 36, 7),
        "octaphene": (34, 41, 8),
        "nonaphene": (38, 46, 9),
        "decaphene": (42, 51, 10),
    }
    assert {
        name: (len(generated[name].atoms), len(generated[name].bonds), len(generated[name].rings))
        for name in expected_sizes
    } == expected_sizes
    assert all(template.numbering_policy == "generated_polyaphene_series" for template in generated.values())


def test_hydrogenated_parent_uses_data_backed_preferred_name():
    policy = retained_parent_name_policy("indoline")
    assert policy is not None
    assert policy.preferred_name == "2,3-dihydro-1H-indole"
    assert policy.output_name("unsubstituted_parent") == policy.preferred_name
    assert policy.output_name("composite_parent") == "indoline"
    assert policy.hydrogenation is not None
    assert policy.hydrogenation.base_parent == "1H-indole"
    assert policy.hydrogenation.hydro_locants == ("2", "3")
    assert name_smiles("c1ccc2c(c1)CCN2") == policy.preferred_name


def test_hydrogenated_hydrocarbon_uses_data_backed_preferred_name():
    policy = retained_parent_name_policy("indane")
    assert policy is not None
    assert policy.accepted_names == ("2,3-dihydro-1H-indene", "indane", "indan")
    assert policy.output_name("unsubstituted_parent") == "2,3-dihydro-1H-indene"
    assert policy.output_name("composite_parent") == "indane"
    assert name_smiles("c1ccc2c(c1)CCC2") == policy.preferred_name


def test_retained_parent_policy_separates_preferred_name_from_accepted_spelling():
    policy = retained_parent_name_policy("dibenz[a,h]anthracene")
    assert policy is not None
    assert policy.preferred_name == "dibenz[a,h]anthracene"
    assert "dibenzo[a,h]anthracene" in policy.accepted_names
    template = next(template for template in retained_fused_graph_templates() if template.name == policy.template_name)
    assert template.output_name == policy.preferred_name
    assert "dibenzo[a,h]anthracene" in template.aliases


def _require_opsin() -> None:
    if py2opsin is None:
        pytest.skip("py2opsin is not available")
    if shutil.which("java") is None:
        pytest.skip("Java runtime not found (OPSIN requires Java)")


def _opsin(name: str, output_format: str = "SMILES") -> str:
    _require_opsin()
    with tempfile.TemporaryDirectory(prefix="openclatura_fused_opsin_") as tmpdir:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return py2opsin.py2opsin(name, output_format=output_format, tmp_fpath=f"{tmpdir}/input.txt")


@pytest.mark.opsin
@pytest.mark.parametrize("parent_name", NEW_TEMPLATE_NAMES)
def test_new_fused_template_locants_and_edges_equal_opsin_cml(parent_name):
    template = next(template for template in retained_fused_graph_templates() if template.name == parent_name)
    root = ET.fromstring(_opsin(parent_name, output_format="CML"))
    namespace = {"cml": "http://www.xml-cml.org/schema"}
    locants_by_id = {}
    for atom in root.findall(".//cml:atom", namespace):
        labels = atom.findall("cml:label", namespace)
        locants_by_id[atom.get("id", "")] = next(
            (label.get("value") for label in labels if label.get("value", "")[:1].isdigit()),
            None,
        )
    edges = set()
    for bond in root.findall(".//cml:bond", namespace):
        left, right = bond.get("atomRefs2", "").split()
        if locants_by_id[left] is not None and locants_by_id[right] is not None:
            edges.add(frozenset((locants_by_id[left], locants_by_id[right])))

    assert set(template.locants) == {locant for locant in locants_by_id.values() if locant is not None}
    assert {frozenset(bond.locants) for bond in template.bonds} == edges


@pytest.mark.opsin
def test_requested_fused_names_roundtrip_through_opsin_one_by_one():
    expected_smiles = [_opsin(source_name) for source_name, _, _ in FUSED_HYDROCARBON_CASES]
    generated = [result.name for result in name_many(expected_smiles, processes=1)]
    roundtripped = [_opsin(name) for name in generated]
    assert all(
        standardize_mol(original) == standardize_mol(back)
        for original, back in zip(expected_smiles, roundtripped, strict=True)
    )


@pytest.mark.opsin
def test_on_demand_tridecacene_roundtrips_through_opsin():
    original = _opsin("tridecacene")
    generated = name_smiles(original)
    assert generated == "tridecacene"
    assert standardize_mol(_opsin(generated)) == standardize_mol(original)

    derivative = _opsin("1-methyltridecacene")
    derivative_name = name_smiles(derivative)
    assert derivative_name == "1-methyltridecacene"
    assert standardize_mol(_opsin(derivative_name)) == standardize_mol(derivative)

    angular = _opsin("tridecaphene")
    angular_name = name_smiles(angular)
    assert angular_name == "tridecaphene"
    assert standardize_mol(_opsin(angular_name)) == standardize_mol(angular)


@pytest.mark.opsin
@pytest.mark.parametrize(
    "derivative_name",
    (
        "1-methyltetracene",
        "1-methylhexacene",
        "1-hydroxybenz[a]anthracene",
        "6-methylbenzo[a]pyrene",
        "1-methylbenzo[b]fluoranthene",
        "1-methylindeno[1,2,3-cd]pyrene",
        "1-methylbenzo[ghi]perylene",
    ),
)
def test_new_parent_locant_maps_support_derivatives(derivative_name):
    original = _opsin(derivative_name)
    generated = name_smiles(original)
    assert standardize_mol(_opsin(generated)) == standardize_mol(original)
