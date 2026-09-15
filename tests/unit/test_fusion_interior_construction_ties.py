"""Construction resolves peripheral symmetry without relaxing graph identity."""

import xml.etree.ElementTree as ET
from dataclasses import fields, replace
from random import Random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion import numbering, planner
from openclatura.fusion.citation_numbering import numbered_parent_graphs_agree
from openclatura.fusion.model import FusedLayout, FusionConfirmed, FusionUnsupported
from openclatura.graph_io import read_rdkit_mol

SMILES = "C1=CC=C2C=CC3=CC=CC4=CC=C1C2=C34"
BASE = "benzo[1,2,3,4-def]phenanthrene"


def _plan(graph):
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    return mol, atoms, planner.plan_fusion_parent(mol, atoms, mode="general")


def _orders(graph):
    order = list(range(graph.GetNumAtoms()))
    yield order
    yield order[::-1]
    for seed in (7, 73, 20483):
        shuffled = order[:]
        Random(seed).shuffle(shuffled)
        yield shuffled


def _labelled_opsin_graph(tmp_path):
    from py2opsin import py2opsin

    root = ET.fromstring(py2opsin(BASE, output_format="CML", tmp_fpath=str(tmp_path / "parent.txt")))
    ns = {"c": "http://www.xml-cml.org/schema"}
    labels = {
        atom.attrib["id"]: atom.find("c:label", ns).attrib["value"]
        for atom in root.findall(".//c:atom", ns)
        if atom.attrib["elementType"] != "H"
    }
    edges = {
        frozenset(labels[atom] for atom in bond.attrib["atomRefs2"].split())
        for bond in root.findall(".//c:bond", ns)
        if all(atom in labels for atom in bond.attrib["atomRefs2"].split())
    }
    assert len(labels) == 16
    return set(labels.values()), edges


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("substitution", (None, (0, "C", "methyl"), (9, "Cl", "chloro")))
def test_every_selected_interior_map_matches_raw_opsin_graph(substitution, tmp_path):
    graph = Chem.RWMol(Chem.MolFromSmiles(SMILES))
    if substitution:
        parent, symbol, _ = substitution
        graph.AddBond(parent, graph.AddAtom(Chem.Atom(symbol)), Chem.BondType.SINGLE)
        Chem.SanitizeMol(graph)
    expected_vertices, expected_edges = _labelled_opsin_graph(tmp_path)
    for order in _orders(graph):
        mol, atoms, result = _plan(Chem.RenumberAtoms(graph, order))
        assert isinstance(result, FusionConfirmed), result
        plan = result.plan
        assert plan.audit.confirmed
        assert plan.rendered_base_name == BASE
        maps = plan.numbering.string_input_locant_maps()
        assert len(maps) == 2
        assert numbered_parent_graphs_agree(mol, maps)
        assert any("stable construction enumeration" in item.reason for item in plan.numbering.rejected_numberings)
        for mapping in maps:
            assert set(mapping.values()) == expected_vertices
            assert {
                frozenset((mapping[bond.u], mapping[bond.v]))
                for bond in mol.bonds.values()
                if bond.u in atoms and bond.v in atoms
            } == expected_edges
            name = BASE
            if substitution:
                parent, _, prefix = substitution
                name = f"{mapping[order.index(parent)]}-{prefix}{BASE}"
            check = verify_with_opsin(name, Chem.MolToSmiles(graph), standardize_smiles=False)
            assert check.status == "matched", check.to_dict()
            assert check.canonical_original == check.canonical_roundtrip


def test_unresolved_interior_maps_still_hit_the_existing_guard(monkeypatch):
    monkeypatch.setattr(numbering, "_construction_ordered_numberings", lambda mol, layouts, candidates: candidates)
    _, _, result = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(result, FusionUnsupported)
    assert "tied fusion numberings identify different locant-labelled parent graphs" in result.details


def test_construction_provenance_is_required_to_resolve_the_tie(monkeypatch):
    original = numbering._construction_ordered_numberings
    checked = []

    def verify(mol, layouts, candidates):
        result = original(mol, layouts, candidates)
        if result != candidates:
            physical = tuple(
                FusedLayout(**{field.name: getattr(item, field.name) for field in fields(FusedLayout)})
                for item in layouts
            )
            assert original(mol, physical, candidates) == candidates
            unranked = tuple(replace(item, construction_numbering_priority=None) for item in layouts)
            with monkeypatch.context() as context:

                def unexpected_signature(*args):
                    pytest.fail("unranked layouts must not repeat graph-signature work")

                context.setattr(numbering, "numbered_parent_graphs_agree", unexpected_signature)
                assert original(mol, unranked, candidates) == candidates
            tied_priority = tuple(replace(item, construction_numbering_priority=0) for item in layouts)
            assert original(mol, tied_priority, candidates) == candidates
            assert not numbered_parent_graphs_agree(mol, (dict(item.atom_to_locant) for item in candidates))
            checked.append(True)
        return result

    monkeypatch.setattr(numbering, "_construction_ordered_numberings", verify)
    _, _, result = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(result, FusionConfirmed)
    assert checked


def test_public_retained_parent_precedence_is_unchanged():
    result = name_mol(Chem.MolFromSmiles(SMILES))
    assert result.error is None
    assert result.name == "pyrene"


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("parent", ("perylene", "benzo[a]pyrene", "benzo[e]pyrene", "benzo[ghi]perylene"))
def test_neighboring_hexagonal_parents_match_every_raw_labelled_map(parent, tmp_path):
    from py2opsin import py2opsin

    from openclatura.retained_fused_templates import retained_graph_templates

    template = next(template for template in retained_graph_templates(include_disabled=True) if template.name == parent)
    graph = Chem.RWMol()
    ids = {}
    for atom in template.atoms:
        rd_atom = Chem.Atom(atom.symbol)
        rd_atom.SetIsAromatic(True)
        ids[atom.locant] = graph.AddAtom(rd_atom)
    for bond in template.bonds:
        graph.AddBond(*(ids[locant] for locant in bond.locants), Chem.BondType.AROMATIC)
    Chem.SanitizeMol(graph)
    expected_name = None
    for order in _orders(graph):
        mol, atoms, result = _plan(Chem.RenumberAtoms(graph, order))
        assert isinstance(result, FusionConfirmed), result
        name = result.plan.rendered_base_name
        if expected_name is None:
            expected_name = name
        assert name == expected_name
        root = ET.fromstring(py2opsin(name, output_format="CML", tmp_fpath=str(tmp_path / "neighbor.txt")))
        ns = {"c": "http://www.xml-cml.org/schema"}
        labels = {
            atom.attrib["id"]: atom.find("c:label", ns).attrib["value"]
            for atom in root.findall(".//c:atom", ns)
            if atom.attrib["elementType"] != "H"
        }
        expected = {
            frozenset(labels[atom] for atom in bond.attrib["atomRefs2"].split())
            for bond in root.findall(".//c:bond", ns)
            if all(atom in labels for atom in bond.attrib["atomRefs2"].split())
        }
        for mapping in result.plan.numbering.string_input_locant_maps():
            assert set(mapping.values()) == set(labels.values())
            assert {
                frozenset((mapping[bond.u], mapping[bond.v]))
                for bond in mol.bonds.values()
                if bond.u in atoms and bond.v in atoms
            } == expected
        check = verify_with_opsin(name, Chem.MolToSmiles(graph), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


def test_exhausted_construction_proof_preserves_ambiguity(monkeypatch):
    from openclatura.fusion import construction_order

    original = construction_order.ordered_hexagonal_construction
    checked = []

    def exhausted(ast, specs, model, positions, search_budget):
        result = original(ast, specs, model, positions, 0)
        assert result is None
        checked.append(True)
        return result

    monkeypatch.setattr(construction_order, "ordered_hexagonal_construction", exhausted)
    _, _, result = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(result, FusionUnsupported)
    assert checked


def test_generated_graph_without_construction_provenance_preserves_ambiguity(monkeypatch):
    from openclatura.fusion import construction_order

    original = construction_order.ordered_hexagonal_construction

    def generated(ast, specs, model, positions, search_budget):
        specs = {
            occurrence: replace(
                spec,
                template=replace(spec.template, numbering_policy="generated_acene_series"),
                construction_order=None,
            )
            for occurrence, spec in specs.items()
        }
        result = original(ast, specs, model, positions, search_budget)
        assert result is None
        return result

    monkeypatch.setattr(construction_order, "ordered_hexagonal_construction", generated)
    _, _, result = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(result, FusionUnsupported)


@pytest.mark.parametrize("seed", (7, 73))
@pytest.mark.parametrize("parent", ("pyrene", "benzo[ghi]perylene"))
def test_graph_record_reordering_preserves_construction_witness(seed, parent, monkeypatch):
    from openclatura.fusion import construction_order
    from openclatura.retained_fused_templates import retained_graph_templates

    original = construction_order.ordered_hexagonal_construction
    checked = []

    def reordered(ast, specs, model, positions, search_budget):
        expected = original(ast, specs, model, positions, search_budget)
        changed = {}
        for occurrence, spec in specs.items():
            atoms = list(spec.template.atoms)
            bonds = [replace(bond, locants=tuple(reversed(bond.locants))) for bond in spec.template.bonds]
            Random(seed).shuffle(atoms)
            Random(seed).shuffle(bonds)
            changed[occurrence] = replace(
                spec,
                template=replace(spec.template, atoms=tuple(atoms), bonds=tuple(bonds), locants=spec.locants[::-1]),
            )
        actual = original(ast, changed, model, positions, search_budget)
        assert actual == expected
        if actual is not None:
            checked.append(True)
        return actual

    monkeypatch.setattr(construction_order, "ordered_hexagonal_construction", reordered)
    template = next(template for template in retained_graph_templates(include_disabled=True) if template.name == parent)
    graph = Chem.RWMol()
    ids = {}
    for atom in template.atoms:
        rd_atom = Chem.Atom(atom.symbol)
        rd_atom.SetIsAromatic(True)
        ids[atom.locant] = graph.AddAtom(rd_atom)
    for bond in template.bonds:
        graph.AddBond(*(ids[locant] for locant in bond.locants), Chem.BondType.AROMATIC)
    Chem.SanitizeMol(graph)
    _, _, result = _plan(graph)
    assert isinstance(result, FusionConfirmed)
    if parent == "pyrene":
        assert result.plan.rendered_base_name == BASE
    assert checked


def test_missing_construction_metadata_preserves_ambiguity(monkeypatch):
    from openclatura.fusion import construction_order

    original = construction_order.ordered_hexagonal_construction

    def missing(ast, specs, model, positions, search_budget):
        specs = {occurrence: replace(spec, construction_order=None) for occurrence, spec in specs.items()}
        assert original(ast, specs, model, positions, search_budget) is None
        return None

    monkeypatch.setattr(construction_order, "ordered_hexagonal_construction", missing)
    _, _, result = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(result, FusionUnsupported)


@pytest.mark.parametrize("defect", ("missing_edge", "extra_vertex", "missing_source", "missing_version"))
def test_invalid_component_construction_metadata_is_rejected(defect):
    from openclatura.fusion.registry import fusion_component_registry

    spec = next(item.spec for item in fusion_component_registry().components if item.spec.key == "phenanthrene")
    order = spec.construction_order
    assert order is not None
    with pytest.raises(ValueError):
        if defect == "missing_edge":
            order = replace(order, directed_bond_locants=order.directed_bond_locants[:-1])
        elif defect == "extra_vertex":
            order = replace(order, atom_locants=(*order.atom_locants, "100"))
        elif defect == "missing_source":
            order = replace(order, source="")
        else:
            order = replace(order, version="")
        replace(spec, construction_order=order)


def test_registry_construction_provenance_is_explicit_and_immutable():
    from dataclasses import FrozenInstanceError

    from openclatura.fusion.registry import fusion_component_registry

    certified = {}
    for component in fusion_component_registry().components:
        spec = component.spec
        order = spec.construction_order
        if order is None:
            continue
        assert order.covers(spec.template)
        assert order.version == "2.9.0"
        assert order.source.endswith("/fusionComponents.xml")
        certified[spec.key] = order
    assert set(certified) == {"benzene", "naphthalene", "anthracene", "phenanthrene"}
    with pytest.raises(FrozenInstanceError):
        certified["benzene"].source = "changed"
