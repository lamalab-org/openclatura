"""Component-local hydrogen labels must not leak into fusion citations."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura.fusion.descriptor import (
    _parent_component_name,
    build_fusion_name_ast,
    render_fusion_name,
    render_fusion_name_parts,
)
from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol, read_smiles
from openclatura.opsin_verify import verify_with_opsin

TARGET = "C1=CC2=C3C4=C(C2C=C1)C1C=CC=CC1=C4c1ccccc13"
BASE_NAME = "dibenzo[1',2':1,2;1'',2'':5,6]pentaleno[3,3a,4-ab]indene"
BASE_SMILES = "c1ccc2c(c1)C1=C3C2=c2ccccc2=C3c2ccccc21"


@pytest.mark.parametrize(
    "parent_name,locants,expected",
    [
        ("1H-indene", ("1",), "indene"),
        ("2H-pyran", ("2",), "pyran"),
        ("1H,3H-parent", ("1", "3"), "parent"),
        ("pyran", ("2",), "pyran"),
        ("[1,3]thiazole", (), "[1,3]thiazole"),
        ("2H-parent", ("1",), "2H-parent"),
        ("1H-parent", (), "1H-parent"),
    ],
)
def test_parent_citation_uses_declared_template_prefix_only(parent_name, locants, expected):
    original = fusion_component_registry().get("1H-indene").spec
    template = replace(original.template, default_indicated_h=locants)
    spec = replace(original, parent_name=parent_name, template=template)

    assert _parent_component_name(spec) == expected
    assert spec.parent_name == parent_name
    assert spec.template is template
    assert template.default_indicated_h == locants
    assert template.atoms == original.template.atoms
    assert template.bonds == original.template.bonds


@pytest.fixture(scope="module")
def indene_citations():
    graph = Chem.MolFromSmiles(TARGET)
    result = []
    for ordered in (graph, Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))):
        mol = read_rdkit_mol(ordered)
        planned = plan_fusion_parent(mol, mol.atoms, mode="audited_pin")
        assert isinstance(planned, FusionConfirmed), planned
        result.append((mol, planned.plan.ast))
    return result


def test_indene_fusion_citation_and_bindings_are_atom_order_invariant(indene_citations):
    registry = fusion_component_registry()
    for mol, ast in indene_citations:
        parts = render_fusion_name_parts(ast, registry, mol=mol)
        assert "".join(part.text for part in parts) == BASE_NAME
        parent = next(part for part in parts if part.grammar_role == "fusion_parent_component")
        assert parent.text == "indene"
        assert parent.atom_ids and parent.bond_ids
        assert parent.source == "fusion_renderer"
        assert set().union(*(set(part.atom_ids) for part in parts)) == set(mol.atoms)
        assert set().union(*(set(part.bond_ids) for part in parts)) == set(mol.bonds)
    spec = registry.get("1H-indene").spec
    assert spec.parent_name == "1H-indene"
    assert spec.template.default_indicated_h == ("1",)


@pytest.mark.parametrize("locants", [("1",), ("1", "3")])
def test_local_h_removal_preserves_every_rendered_token_binding(indene_citations, locants):
    mol, ast = indene_citations[0]
    specs = {spec.key: spec for spec in fusion_component_registry().specs}
    key = "1H-indene"
    parent = specs[key]
    baseline = render_fusion_name_parts(ast, specs, mol=mol)
    specs[key] = replace(
        parent,
        parent_name=",".join(f"{locant}H" for locant in locants) + "-indene",
        template=replace(parent.template, default_indicated_h=locants),
    )

    rendered = render_fusion_name_parts(ast, specs, mol=mol)
    assert rendered == baseline
    assert "".join(part.text for part in rendered) == BASE_NAME
    assert all("H" not in part.text for part in rendered)
    assert specs[key].template.default_indicated_h == locants


@pytest.mark.parametrize("style,ending", [("basic", "difuran"), ("complex", "bis(furan)")])
def test_multiparent_citations_use_the_same_component_h_scope(style, ending):
    mol = read_smiles("O1C=2C(C=C1)=CC=1OC=CC1C2")
    registry = fusion_component_registry()
    bounded = select_bounded_face_model(mol, mol.atoms)
    matches = tuple(match for match in registry.match_faces(mol, bounded) if len(match.covered_face_ids) == 1)
    ast = build_fusion_name_ast(mol, matches, registry)
    assert ast.plan_kind == "multiparent"
    specs = {spec.key: spec for spec in fusion_component_registry().specs}
    parent = specs["furan"]
    specs["furan"] = replace(
        parent,
        parent_name="2H-furan",
        template=replace(parent.template, default_indicated_h=("2",)),
        multiplicative_prefix_style=style,
    )

    assert render_fusion_name(ast, specs) == f"benzo[1,2-b:4,5-b']{ending}"
    assert specs["furan"].template.default_indicated_h == ("2",)


@pytest.mark.opsin
def test_rendered_indene_fusion_parent_has_exact_opsin_skeleton(indene_citations):
    for _mol, ast in indene_citations:
        name = render_fusion_name(ast, fusion_component_registry())
        check = verify_with_opsin(name, BASE_SMILES, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.opsin
def test_indene_dihydro_composition_has_exact_opsin_witness():
    check = verify_with_opsin("4a,4c-dihydro" + BASE_NAME, TARGET, standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_roundtrip == check.canonical_original
