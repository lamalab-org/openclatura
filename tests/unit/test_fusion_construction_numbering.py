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
    (
        11029,
        "CC[C@@H]1CN2CC[C@@]34c5ccccc5N5C(=O)CC=C([C@H]1C[C@@H]23)[C@@H]54",
        "pyrido[3,2,1-jk]pyrrolo[3,2-f]carbazole",
    ),
    (11954, "Clc1ccc2c(c1)C1SC2c2c1n1c3ccccc3c3cccc2c31", "indolo[3,2,1-jk]benzo[b]carbazole"),
    (12963, "COc1ccc2c3c1O[C@H]1CCC=C[C@@]31CCN(CC=C(C)C)C2", "benzo[1',2':2,3]benzofuro[4,3a,3-cd]azepine"),
    (
        19452,
        "CC1CCc2cccc3c2N1C(=O)c1cc(NC(=S)Nc2ccccc2)ccc1O3",
        "benzo[1',2':6,7][1,4]oxazepino[2,3,4-ij]quinoline",
    ),
    (
        20380,
        "CN1c2cc(C#N)ccc2N2CC[C@@H](NC(=O)C(F)(F)F)C[C@@H]2c2c(C#N)cccc21",
        "pyrido[1,2-d]dibenzo[b,f][1,4]diazepine",
    ),
    (28506, "O=C1C=C[C@@H]2[C@H]3c4cccc5cccc(c45)[C@H]3[C@H]1N2c1ccccc1", "naphtho[1,8a,8-ab]azulene"),
    (
        31042,
        "COc1ccc(CN2c3ccccc3[C@@]34CCN5C=C[C@@H]6OCC[C@]6(CC[C@H]23)[C@H]54)cc1",
        "indolo[2,3-h]pyrrolo[3,2,1-ij]furo[2,3-d]quinoline",
    ),
    (45607, "CC1=C[C@H]2C[C@H](C)[C@H]3CC[C@H](C)C4=C3[C@@H](OC4=O)[C@@]2(C)C1", "azuleno[4,5,6-cd]2-benzofuran"),
    (52973, "O=C1C=CC(=O)C2=C3C1=CC=CC3N1C=CCN21", "pyrazolo[1,2-a]cyclohepta[cd]indazole"),
    (53931, "N#CC(C#N)=C1c2cc(F)ccc2-c2cc3c(cc21)C(=C(C#N)C#N)C1C=C(F)C=CC31", "indeno[1,2-f]benzo[b]indene"),
    (59172, "CCO[C@@H]1C=C2[C@H](O)CN3CCCc4cc(OC)c(OC)cc4[C@]23C[C@H]1OC", "indolo[1,7a-a]benzo[c]azepine"),
    (
        59728,
        "CC(C)C1=C2[C@H]3CC=C4[C@@H]5[C@@H](O[C@@H]6OC[C@@](O)(C(=O)[C@@]65O)[C@@H]4O)[C@]3(C)CC[C@]2(C)CC1",
        "cyclopenta[1'',2'':1',2']benzo[3',4':1,2]cyclohepta[3,4,5-cd]2-benzofuran",
    ),
    (
        64746,
        "COc1ccc2c(c1)C13CCNC1C1c4[nH]c5ccc(OC)cc5c4CCN1C3N2",
        "benzo[1',2':2,3]pyrrolo[4,5-d]indolo[2',3':2,3]pyrrolo[3',2':3,4]pyrrolo[5,1-f]pyridine",
    ),
    (67508, "CNCCC(=O)N1c2ccccc2N2CCc3cccc(c32)C1C", "[1,5]benzodiazepino[3,2,1-hi]indole"),
    (75694, "O=C(O)N1CCc2c(n3c4c(cccc24)CCC3)CC1", "azepino[4',5':2,3]pyrrolo[4,5,1-ij]quinoline"),
    (78522, "c1ccc(-c2ccccc2-c2c3ccccc3cc3c2c2cccc4c5ccccc5n3c42)cc1", "indolo[3,2,1-jk]benzo[b]carbazole"),
    (
        81443,
        "C[C@@H]1C(=O)O[C@@H]2[C@H]1[C@@]13O[C@@H]4OC(=O)[C@H](O)[C@@]45[C@H](C(C)(C)C)[C@@H](O)[C@@H](OC1=O)[C@]53[C@H]2O",
        "furo[2,3-b]cyclopenta[c]furo[2',3':1,2]cyclopenta[4,3-d]furan",
    ),
    (
        81754,
        "COC1=CC23CCCN2CCc2cc4c(cc2C3(O)C1O)OCO4",
        "pyrrolo[1,2-a][1,3]dioxolo[4',5':1,2]benzo[4,5-d]cyclopenta[b]azepine",
    ),
    (88969, "COc1ccc2c3c1OC1C[C@@H](OC(=O)c4ccc(C(C)(C)C)cc4)C=C[C@@]31CCN2C", "benzofuro[3a,3,2-de]quinoline"),
    (89309, "CN1CCN(C2=Nc3cc(Cl)cc4ccn(c34)-c3ccccc32)CC1", "pyrrolo[1,2,3-ef]benzo[c]1,5-benzodiazepine"),
    (90198, "COc1ccc2c3c1OC1C[C@@H](OC(=O)c4ccc(Cl)cc4Cl)C=C[C@@]31CCN2C", "benzofuro[3a,3,2-de]quinoline"),
    (
        98468,
        "Fc1ccc(-c2ccc3[nH]c4c(c3c2)-c2cccc3cccc-4c23)cc1",
        "benzo[b]naphtho[1',8a',8':1,2,3]cyclopenta[4,5-d]pyrrole",
    ),
    (
        30704,
        "COc1ccc2c(c1)-c1nnc(COc3ccc(F)cc3)n1Cc1c(-c3noc(C)n3)ncn1-2",
        "[1,2,4]triazolo[4,3-d]imidazo[1,5-a]benzo[f][1,4]diazepine",
    ),
    (
        49858,
        "COCc1nnc2n1Cc1c(C3=N[C@H](C)CO3)ncn1-c1ccc(Cl)cc1-2",
        "[1,2,4]triazolo[4,3-d]imidazo[1,5-a]benzo[f][1,4]diazepine",
    ),
    (
        52446,
        "CC(C)(Oc1cccnc1)c1nc(-c2ncn3c2Cn2ncnc2-c2cc(F)ccc2-3)no1",
        "[1,2,4]triazolo[1,5-d]imidazo[1,5-a]benzo[f][1,4]diazepine",
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
    if not atoms:
        cml = py2opsin("perhydro" + base, output_format="CML", tmp_fpath=str(tmp_path / "saturated.txt"))
        root = ET.fromstring(cml)
        atoms = root.findall(".//c:atom", ns)
    assert atoms, f"OPSIN did not provide a labelled skeleton for {base}"
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
            # Symmetric component embeddings may reorder peripheral construction
            # atoms. Only the surviving interior order participates in numbering.
            constructions[-1].add(
                tuple(
                    indices[atom]
                    for atom in witness.construction_atom_order
                    if atom not in proof.selected_face_model.outer_boundary
                )
            )
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
        assert all(2 * solved[index][axis] == solved[index - 1][axis] + solved[index + 1][axis] for axis in (0, 1))
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
    index = 32572
    smiles = next(smiles for case, smiles, _ in CASES if case == index)
    result = name_mol(Chem.MolFromSmiles(smiles), fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)
    assert result.error is None
    orientation = next(step for step in result.decisions if step.decision == "selected preferred fusion orientation")
    assert OPSIN_CONSTRUCTION_NUMBERING in orientation.data["audit_evidence"]
    assert any("PIN not certified" in evidence for evidence in orientation.data["audit_evidence"])


def test_entry_compatibility_witness_projects_through_existing_plan_trace(monkeypatch):
    from openclatura.fusion.trace import trace_confirmed_fusion_plan
    from openclatura.molecule import DecisionTrace

    _, smiles, base = next(case for case in CASES if case[0] == 67508)
    plans = _capture_plans(monkeypatch, base)
    graph = Chem.MolFromSmiles(smiles)
    assert name_mol(graph, fusion_mode=FusionMode.AUDITED_PIN).error is None
    plan = plans[0]
    trace = DecisionTrace()
    trace_confirmed_fusion_plan(trace, read_rdkit_mol(graph), plan, dict(plan.numbering.input_locant_maps[0]))
    orientation = next(step for step in trace.steps if step.decision == "selected preferred fusion orientation")
    assert OPSIN_CONSTRUCTION_NUMBERING in orientation.data["audit_evidence"]
    assert any("PIN not certified" in evidence for evidence in orientation.data["audit_evidence"])


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
def test_partial_construction_coverage_does_not_break_a_previously_passing_parent():
    smiles = "CC[C@]12CCC3C4=C(C=C(N)CC4)C4(CC4)CC3C1C1CC1[C@@]21CCC(=O)O1"
    result = name_mol(Chem.MolFromSmiles(smiles), fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None
    assert verify_with_opsin(result.name, smiles, standardize_smiles=False).status == "matched"


def test_entry_witness_rejects_invalid_perimeters_and_ring_positions(monkeypatch):
    from openclatura.fusion.layout import OpsinEntryLayout

    _, smiles, base = next(case for case in CASES if case[0] == 67508)
    plans = _capture_plans(monkeypatch, base)
    graph = Chem.MolFromSmiles(smiles)
    assert name_mol(graph, fusion_mode=FusionMode.AUDITED_PIN).error is None
    plan = plans[0]
    witness = plan.numbering.selected_layout
    assert isinstance(witness, OpsinEntryLayout)
    with pytest.raises(ValueError, match="injective"):
        replace(witness, entry_perimeter=witness.entry_perimeter + witness.entry_perimeter[:1])
    with pytest.raises(ValueError, match="every positioned face"):
        replace(witness, entry_ring_positions=witness.entry_ring_positions[:-1])
    invalid = list(witness.entry_perimeter)
    invalid[1], invalid[3] = invalid[3], invalid[1]
    mol = read_rdkit_mol(graph)
    bounded = select_bounded_face_model(mol, dict(plan.numbering.abstract_atom_to_locant))
    assert (
        _numbering_from_layout(
            mol,
            bounded,
            plan.numbering.selected_face_model,
            replace(witness, entry_perimeter=tuple(invalid)),
            0,
            set(bounded.fusion_atoms),
        )
        is None
    )


def test_entry_direction_search_is_bounded_and_cached(monkeypatch):
    from openclatura.fusion.entry_geometry import entry_direction_geometry
    from openclatura.fusion.layout import LayoutSearchBudgetExceeded

    _, smiles, base = next(case for case in CASES if case[0] == 67508)
    plans = _capture_plans(monkeypatch, base)
    assert name_mol(Chem.MolFromSmiles(smiles), fusion_mode=FusionMode.AUDITED_PIN).error is None
    model = plans[0].numbering.selected_face_model
    with pytest.raises(LayoutSearchBudgetExceeded):
        entry_direction_geometry(model, 1)
    first = entry_direction_geometry(model, 100000)
    hits = entry_direction_geometry.cache_info().hits
    assert entry_direction_geometry(model, 100000) is first
    assert entry_direction_geometry.cache_info().hits == hits + 1


@pytest.mark.parametrize(
    "size,ports,expected",
    (
        (5, {1}, ((-2, 0, 1, 3),)),
        (5, {0, 1, 4}, ((-3, -1, 1, 3),)),
        (5, {0, 2}, ((-2, 0, 1, 3), (-3, -1, 0, 2))),
        (5, {0, 3}, ((-2, 0, 1, 3), (-3, -1, 0, 2))),
        (5, {0, 1}, ((-3, -1, 0, 2),)),
        (5, {0, 1, 2, 3, 4}, ((-2, 0, 1, 3), (-3, -1, 0, 2), (-3, -1, 1, 3))),
        (7, {0, 2}, ((-3, -1, 0, 1, 2, 3), (-3, -2, -1, 1, 2, 3))),
    ),
)
def test_parser_shape_admission_preserves_order_and_removes_port_degeneracy(size, ports, expected):
    from openclatura.fusion.entry_geometry import _allowed_entry_directions

    assert _allowed_entry_directions(size, frozenset(ports)) == expected


def test_pentagon_top_right_shape_requires_top_left_to_be_disallowed():
    from openclatura.fusion.entry_geometry import _entry_direction_tables

    tables, _ = _entry_direction_tables()
    _, top_left_forbidden, _ = tables[5][2]
    _, _, top_right_required = tables[5][3]
    assert top_left_forbidden == top_right_required == (2,)


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
def test_cited_terminal_entry_tracks_a_graph_variant():
    _, smiles, _ = next(case for case in CASES if case[0] == 30704)
    graph = Chem.RWMol(Chem.MolFromSmiles(smiles))
    next(atom for atom in graph.GetAtoms() if atom.GetAtomicNum() == 9).SetAtomicNum(17)
    Chem.SanitizeMol(graph)
    result = name_mol(graph, fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None
    assert verify_with_opsin(result.name, Chem.MolToSmiles(graph), standardize_smiles=False).status == "matched"
