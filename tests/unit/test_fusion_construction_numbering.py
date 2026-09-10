"""Construction-bound interior locants preserve the complete labelled graph."""

import random
import xml.etree.ElementTree as ET
from dataclasses import fields, replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.fusion import planner
from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.layout import (
    OPSIN_CONSTRUCTION_NUMBERING,
    OpsinConstructionLayout,
    opsin_construction_atom_order,
)
from openclatura.fusion.model import FusedLayout, FusionConfirmed, FusionMode
from openclatura.fusion.numbering import _number_completed_system, _numbering_from_layout
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol
from openclatura.opsin_verify import verify_with_opsin

CASES = (
    (
        31149,
        "O=C(Nc1ccncc1)C1=C[C@@H]2Oc3c(O)ccc4c3[C@@]23CC12CN(CC1CC1)[C@H](C4)[C@]23O",
        "benzo[1',2',3':3,3a,4]isobenzofurano[6,7,7a,1-cde]isoindole",
    ),
    (
        32572,
        "C[C@@]12C(O)C=CC1C1=CC3CCC=C4C=CC5=CCC2C1C5C43",
        "pentaleno[1,2,3-jk]benzo[1,2,3,4-def]phenanthrene",
    ),
    (
        76220,
        "CN=c1c2c(N3CCO[C@@H](COC)C3)c(F)cc3c(=O)c(C(=O)O)c4scc1n4c32",
        "pentaleno[1,6a,6,5-cde]naphthalene",
    ),
    (
        79496,
        "O=c1nc2c3ccccc3nc3sc4cccc1c4n32",
        "benzo[a]cyclopenta[def]phenanthrene",
    ),
)


def _capture_plans(monkeypatch, base):
    original = planner._complete_fusion_plan
    plans = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if isinstance(result, FusionConfirmed) and result.plan.rendered_base_name.endswith(base):
            plans.append(result.plan)
        return result

    monkeypatch.setattr(planner, "_complete_fusion_plan", capture)
    return plans


def _opsin_labelled_graph(base, tmp_path):
    from py2opsin import py2opsin

    cml = py2opsin(base, output_format="CML", tmp_fpath=str(tmp_path / "input.txt"))
    root = ET.fromstring(cml)
    ns = {"c": "http://www.xml-cml.org/schema"}
    atoms = root.findall(".//c:atom", ns)
    labels = {
        atom.attrib["id"]: atom.find("c:label", ns).attrib["value"]
        for atom in atoms
        if atom.find("c:label", ns) is not None
    }
    elements = {labels[atom.attrib["id"]]: atom.attrib["elementType"] for atom in atoms if atom.attrib["id"] in labels}
    edges = {
        frozenset(labels[atom] for atom in bond.attrib["atomRefs2"].split())
        for bond in root.findall(".//c:bond", ns)
        if all(atom in labels for atom in bond.attrib["atomRefs2"].split())
    }
    return elements, edges


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("index,smiles,base", CASES, ids=[str(case[0]) for case in CASES])
def test_exact_structures_and_complete_locant_graphs_under_permutations(index, smiles, base, monkeypatch, tmp_path):
    expected_elements, expected_edges = _opsin_labelled_graph(base, tmp_path)
    graph = Chem.MolFromSmiles(smiles)
    order = list(range(graph.GetNumAtoms()))
    shuffled = order.copy()
    random.Random(73).shuffle(shuffled)
    plans = _capture_plans(monkeypatch, base)
    names, maps, constructions = [], [], []
    for indices in (order, list(reversed(order)), shuffled):
        plans.clear()
        result = name_mol(Chem.RenumberAtoms(graph, indices), fusion_mode=FusionMode.AUDITED_PIN)
        assert result.error is None
        assert result.name.isascii()
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", (index, check.to_dict())
        assert check.canonical_original == check.canonical_roundtrip
        assert plans
        names.append(result.name)
        maps.append(set())
        constructions.append(set())
        for plan in plans:
            proof = plan.numbering
            witness = proof.selected_layout
            assert isinstance(witness, OpsinConstructionLayout)
            assert OPSIN_CONSTRUCTION_NUMBERING in witness.audit_evidence
            locants = dict(proof.abstract_atom_to_locant)
            assert set(locants) == {atom.id for atom in plan.abstract_parent_graph.atoms}
            assert len(set(locants.values())) == len(locants)
            assert all(locant.interior_distance is None for locant in locants.values())
            assert {
                str(locants[atom.id]): atom.symbol for atom in plan.abstract_parent_graph.atoms
            } == expected_elements
            assert {
                frozenset(str(locants[atom]) for atom in bond.atoms) for bond in plan.abstract_parent_graph.bonds
            } == expected_edges
            assert set(witness.construction_atom_order) == set(locants)
            maps[-1].add(tuple(sorted((indices[atom], str(locant)) for atom, locant in locants.items())))
            constructions[-1].add(tuple(indices[atom] for atom in witness.construction_atom_order))
    assert names[0] == names[1] == names[2]
    assert maps[0] == maps[1] == maps[2]
    assert constructions[0] == constructions[1] == constructions[2]


@pytest.fixture
def carbon_context(monkeypatch):
    _, smiles, base = CASES[1]
    graph = Chem.MolFromSmiles(smiles)
    plans = _capture_plans(monkeypatch, base)
    result = name_mol(graph, fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None and plans
    plan = plans[0]
    mol = read_rdkit_mol(graph)
    bounded = select_bounded_face_model(mol, dict(plan.numbering.abstract_atom_to_locant))
    assert bounded is not None
    return mol, bounded, plan


def test_compatibility_is_explicit_and_preserves_graph_only_numbering(carbon_context):
    mol, bounded, plan = carbon_context
    proof = plan.numbering
    expected = dict(proof.abstract_atom_to_locant)
    perimeter = tuple(sorted(bounded.outer_boundary.atoms, key=lambda atom: expected[atom]))
    original = _number_completed_system(mol, bounded, perimeter, set(bounded.fusion_atoms))
    compatible = _number_completed_system(
        mol,
        bounded,
        perimeter,
        set(bounded.fusion_atoms),
        opsin_atom_order=proof.selected_layout.construction_atom_order,
    )
    assert compatible == expected
    assert set(original) == set(compatible) == set(bounded.atom_ids)
    assert all(original[atom] == compatible[atom] for atom in perimeter)
    assert all(original[atom].interior_distance == 1 for atom in bounded.interior_atoms)
    assert all(compatible[atom].interior_distance is None for atom in bounded.interior_atoms)


def test_construction_does_not_depend_on_component_or_mapping_storage_order(carbon_context):
    _, _, plan = carbon_context
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    shuffled_ast = replace(
        plan.ast,
        component_occurrences=tuple(
            replace(match, local_to_input_atom=tuple(reversed(match.local_to_input_atom)))
            for match in reversed(plan.ast.component_occurrences)
        ),
    )
    assert opsin_construction_atom_order(shuffled_ast, specs) == plan.numbering.selected_layout.construction_atom_order


def test_unproved_component_overlap_is_rejected(carbon_context):
    _, _, plan = carbon_context
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    join = plan.ast.joins[0]
    target = next(match for match in plan.ast.component_occurrences if match.occurrence_id == join.attached_occurrence)
    local = target.input_atom_by_locant
    a, b = (locant.text for locant in join.interface.attached_path[:2])
    local[a], local[b] = local[b], local[a]
    broken = replace(
        plan.ast,
        component_occurrences=tuple(
            replace(match, local_to_input_atom=tuple(local.items())) if match == target else match
            for match in plan.ast.component_occurrences
        ),
    )
    assert opsin_construction_atom_order(broken, specs) is None


def test_construction_witness_rejects_missing_or_duplicate_atoms(carbon_context):
    _, _, plan = carbon_context
    layout = plan.numbering.selected_layout
    order = layout.construction_atom_order
    for malformed in (order[:-1], order + order[:1], order[:-1] + (max(order) + 1,)):
        with pytest.raises(ValueError, match="construction atom order"):
            replace(layout, construction_atom_order=malformed)


def test_numbering_cache_includes_construction_provenance(carbon_context):
    mol, bounded, plan = carbon_context
    proof = plan.numbering
    layout = proof.selected_layout
    ordinary = FusedLayout(**{field.name: getattr(layout, field.name) for field in fields(FusedLayout)})
    reversed_order = replace(layout, construction_atom_order=tuple(reversed(layout.construction_atom_order)))
    cache = {}
    results = [
        _numbering_from_layout(
            mol, bounded, proof.selected_face_model, candidate, index, set(bounded.fusion_atoms), perimeter_cache=cache
        )
        for index, candidate in enumerate((ordinary, layout, reversed_order, layout))
    ]
    assert all(result is not None for result in results)
    assert len(cache) == 3
    assert results[0].atom_to_locant != results[1].atom_to_locant
    assert results[1].atom_to_locant != results[2].atom_to_locant
    assert results[1].atom_to_locant == results[3].atom_to_locant


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
def test_distinguishable_interior_substituents_roundtrip_without_example_lookup(carbon_context):
    _, bounded, _ = carbon_context
    for atom in bounded.interior_atoms:
        variant = Chem.RWMol(Chem.MolFromSmiles(CASES[1][1]))
        methyl = variant.AddAtom(Chem.Atom(6))
        variant.AddBond(atom, methyl, Chem.BondType.SINGLE)
        Chem.SanitizeMol(variant)
        smiles = Chem.MolToSmiles(variant)
        result = name_mol(variant, fusion_mode=FusionMode.AUDITED_PIN)
        assert result.error is None
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip


def test_compatibility_witness_reaches_the_existing_accepted_orientation_trace():
    result = name_mol(Chem.MolFromSmiles(CASES[1][1]), fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)
    assert result.error is None
    orientation = next(step for step in result.decisions if step.decision == "selected preferred fusion orientation")
    assert OPSIN_CONSTRUCTION_NUMBERING in orientation.data["audit_evidence"]
    assert any("PIN not certified" in evidence for evidence in orientation.data["audit_evidence"])
