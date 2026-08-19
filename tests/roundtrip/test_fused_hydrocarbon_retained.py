"""Graph-backed retained-name coverage for the requested fused hydrocarbons."""

from __future__ import annotations

import shutil
import tempfile
import warnings
import xml.etree.ElementTree as ET

import pytest
from rdkit import Chem

from openclatura import name_many, name_smiles
from openclatura.retained_fused_templates import _generated_acene_templates, retained_fused_graph_templates
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
