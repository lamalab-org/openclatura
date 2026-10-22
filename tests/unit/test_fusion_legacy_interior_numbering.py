"""FR-5.5.2 legacy interior numbering is topological and reproducible."""

import importlib.metadata
import random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.fusion import planner
from openclatura.fusion.model import FusionConfirmed, FusionMode, FusionUnsupported
from openclatura.fusion.registry import FusionComponentRegistry
from openclatura.graph_io import read_rdkit_mol

SMILES = "CC(C)=CCC[C@]1(C)Oc2cc(CO)c(C=O)c3c2[C@H]2[C@@H](O3)[C@@](C)(O)CC[C@H]21"
EXPECTED_NAME = (
    "(3S,3aR,9S,9aR,9bS)-3-hydroxy-6-(hydroxymethyl)-3,9-dimethyl-"
    "9-(4-methylpent-3-en-1-yl)-2,3,3a,4,8,9,9a,9b-octahydro-"
    "4,8-dioxa1H-cyclopenta[def]phenanthrene-5-carbaldehyde"
)
ASSIGNMENT_A = {
    frozenset(("3a", "9b")),
    frozenset(("4a", "9c")),
    frozenset(("7a", "9c")),
    frozenset(("9a", "9b")),
    frozenset(("9b", "9c")),
}
ASSIGNMENT_B = {
    frozenset(("3a", "9c")),
    frozenset(("4a", "9b")),
    frozenset(("7a", "9b")),
    frozenset(("9a", "9c")),
    frozenset(("9b", "9c")),
}


def _interior_edge_signature(mol, mapping):
    locants = {atom: str(locant) for atom, locant in mapping.items()}
    relevant = {"3a", "4a", "7a", "9a", "9b", "9c"}
    edges = set()
    for bond in mol.bonds.values():
        if bond.u not in locants or bond.v not in locants:
            continue
        edge = frozenset((locants[bond.u], locants[bond.v]))
        if edge <= relevant and edge & {"9b", "9c"}:
            edges.add(edge)
    return edges


def _captured_plan(monkeypatch, graph):
    original = planner._complete_fusion_plan
    plans = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if isinstance(result, FusionConfirmed) and result.plan.rendered_base_name.endswith(
            "cyclopenta[def]phenanthrene"
        ):
            plans.append(result.plan)
        return result

    monkeypatch.setattr(planner, "_complete_fusion_plan", capture)
    result = name_mol(graph, fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None and plans
    return plans[0]


def test_assignment_a_distinguishes_the_two_interior_atoms(monkeypatch):
    graph = Chem.MolFromSmiles(SMILES)
    plan = _captured_plan(monkeypatch, graph)
    mol = read_rdkit_mol(graph)
    mapping = dict(plan.numbering.abstract_atom_to_locant)
    assert _interior_edge_signature(mol, mapping) == ASSIGNMENT_A
    assert _interior_edge_signature(mol, mapping) != ASSIGNMENT_B

    by_locant = {str(locant): atom for atom, locant in mapping.items()}
    x = set(mol.get_neighbors(by_locant["3a"])) & set(mol.get_neighbors(by_locant["9a"]))
    y = set(mol.get_neighbors(by_locant["4a"])) & set(mol.get_neighbors(by_locant["7a"]))
    assert x == {by_locant["9b"]}
    assert y == {by_locant["9c"]}


def test_every_equivalent_decomposition_uses_assignment_a(monkeypatch):
    graph = Chem.MolFromSmiles(SMILES)
    mol = read_rdkit_mol(graph)
    selections = []
    original_selection = planner.completed_system_numbering_selection
    original_candidate = planner._plan_numbered_candidate

    def capture_selection(*args, **kwargs):
        result = original_selection(*args, **kwargs)
        selections.append(result)
        return result

    def exhaust_candidates(*args, **kwargs):
        original_candidate(*args, **kwargs)
        return FusionUnsupported("enumerate every equivalent decomposition")

    monkeypatch.setattr(planner, "completed_system_numbering_selection", capture_selection)
    monkeypatch.setattr(planner, "_plan_numbered_candidate", exhaust_candidates)
    name_mol(graph, fusion_mode=FusionMode.AUDITED_PIN)

    maps = [dict(numbering.atom_to_locant) for selection in selections for numbering in selection.accepted]
    assert len(maps) >= 3
    assert all(_interior_edge_signature(mol, mapping) == ASSIGNMENT_A for mapping in maps)


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_complete_name_is_input_and_component_order_invariant(monkeypatch):
    graph = Chem.MolFromSmiles(SMILES)
    order = list(range(graph.GetNumAtoms()))
    shuffled = order.copy()
    random.Random(73).shuffle(shuffled)
    names = []
    for indices in (order, list(reversed(order)), shuffled):
        result = name_mol(Chem.RenumberAtoms(graph, indices), verify_opsin=True)
        assert result.error is None
        assert result.opsin_check.status == "matched", result.opsin_check.to_dict()
        assert result.opsin_check.canonical_original == result.opsin_check.canonical_roundtrip
        names.append(result.name)
    assert names == [EXPECTED_NAME] * 3

    original = FusionComponentRegistry.match_faces

    def reversed_components(self, *args, **kwargs):
        return tuple(reversed(original(self, *args, **kwargs)))

    monkeypatch.setattr(FusionComponentRegistry, "match_faces", reversed_components)
    result = name_mol(graph, verify_opsin=True)
    assert result.name == EXPECTED_NAME
    assert result.opsin_check.status == "matched"


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_pinned_opsin_reconstructs_the_complete_stereochemical_structure():
    assert importlib.metadata.version("py2opsin") == "1.2.0"
    result = name_mol(Chem.MolFromSmiles(SMILES), verify_opsin=True)
    assert result.name == EXPECTED_NAME
    assert result.parent_nomenclature == "skeletal_replacement_fusion"
    assert result.opsin_check.status == "matched", result.opsin_check.to_dict()
    assert result.opsin_check.canonical_original == result.opsin_check.canonical_roundtrip
