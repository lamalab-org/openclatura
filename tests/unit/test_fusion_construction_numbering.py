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
    OPSIN_COUPLED_PENTAGON_AXES,
    OPSIN_RING_MAP_NUMBERING,
    RING_SHAPE_TEMPLATES,
    OpsinConstructionLayout,
    _coupled_pentagon_axis_centers,
    _opsin_occupied_row_orientation,
    _opsin_pentagon_chain,
    _orientation_score,
    opsin_construction_atom_order,
    preferred_intrinsic_layouts,
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
    (
        4582,
        "CCCCCC(=O)O[C@@H]1[C@@]2(C(C)C)O[C@H]2[C@@H]2O[C@]23[C@]12O[C@H]2C[C@H]1C2=C(CC[C@@]13C)C(=O)OC2",
        "trisoxireno[2',3':2,3;2'',3'':4,4a;2''',3''':10,10a]phenanthro[7,8-c]furan",
    ),
    (
        58492,
        "CC(C)[C@]12O[C@H]1[C@@H]1O[C@]13[C@]1(O[C@H]1C[C@H]1C4=C(C(=O)c5ccccc5)OC(=O)C4CC[C@@]13C)[C@@H]2O",
        "trisoxireno[2',3':2,3;2'',3'':4,4a;2''',3''':10,10a]phenanthro[7,8-c]furan",
    ),
    (
        44983,
        "COC1=CC2SC3=C(C(=O)C4CCC=CC34)C2C=C1",
        "indeno[1,2-b]1-benzothiophene",
    ),
    (
        48496,
        "OC1=CC2SC3=C(C2C=C1)C(O)(c1ccc(OCCN2CCCCC2)cc1)c1ccccc13",
        "indeno[1,2-b]1-benzothiophene",
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
            if index in {4582, 58492}:
                assert OPSIN_RING_MAP_NUMBERING in witness.audit_evidence
            if index in {44983, 48496}:
                assert OPSIN_COUPLED_PENTAGON_AXES in witness.audit_evidence
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


def test_ring_map_compatibility_abstains_outside_the_proved_tree_family(carbon_context):
    _, _, plan = carbon_context
    assert preferred_intrinsic_layouts(plan.numbering.selected_face_model, opsin_ring_map=True) == ()


def test_ring_map_quadrants_preserve_the_connected_axis_length():
    centers = {0: (0, 0), 1: (2, 0), 2: (4, 0), 3: (6, 0), 4: (3, 2)}
    adjacent = frozenset(frozenset(pair) for pair in ((0, 1), (1, 4), (4, 2), (2, 3)))
    shape = next(shape for shape in RING_SHAPE_TEMPLATES if shape.ring_size == 6)
    orders = {face: (face,) for face in centers}
    ordinary = _orientation_score(centers, dict.fromkeys(centers, shape), adjacent, orders=orders, positions=centers)
    compatible = _orientation_score(
        centers, dict.fromkeys(centers, shape), adjacent, orders=orders, positions=centers, opsin_ring_map=True
    )
    assert ordinary == (0, -2, -10, 2, -12)
    # Two right-axis rings contribute two units each; the upper bisected
    # ring contributes two more. The connected longest row still has two rings.
    assert compatible == (0, -2, -6, 4, -12)


def test_occupied_rows_preserve_gaps_parity_and_storage_independence():
    centers = {0: (0, 0), 1: (4, 0), 2: (12, 0), 3: (2, 0), 4: (2, 1)}
    expected = _opsin_occupied_row_orientation(centers)
    assert expected == (-7, 3, -12)
    transformed = {100 - face: (x + 7, y - 3) for face, (x, y) in reversed(centers.items())}
    assert _opsin_occupied_row_orientation(transformed) == expected


def test_coupled_pentagon_axis_proof_is_topological_and_reversible():
    orders = {0: (0, 1, 2, 3, 4, 5), 1: (4, 5, 6, 7, 8), 2: (6, 7, 9, 10, 11), 3: (9, 10, 12, 13, 14, 15)}
    adjacent = frozenset(frozenset(pair) for pair in ((0, 1), (1, 2), (2, 3)))
    path = _opsin_pentagon_chain(orders, adjacent)
    assert path == (0, 1, 2, 3)
    reordered = {face: order[2:] + order[:2] for face, order in reversed(orders.items())}
    assert _opsin_pentagon_chain(reordered, adjacent) == tuple(reversed(path))
    centers = {0: (1, 2), 1: (5, 99), 2: (-3, 15), 3: (11, 7)}
    solved = _coupled_pentagon_axis_centers(path, centers)
    assert solved == _coupled_pentagon_axis_centers(tuple(reversed(path)), centers)
    for index in (1, 2):
        assert all(2 * solved[index][axis] == solved[index-1][axis] + solved[index+1][axis] for axis in (0, 1))
    assert _opsin_pentagon_chain(orders, adjacent | {frozenset((0, 3))}) is None
    assert _opsin_pentagon_chain({**orders, 3: (7, 9, 12, 13, 14, 15)}, adjacent) is None


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
