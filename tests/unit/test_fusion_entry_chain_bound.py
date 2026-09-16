"""Deep entry chains retain exact geometry under a proved ranking bound."""

from random import Random

import pytest
from rdkit import Chem

from openclatura import opsin_available, verify_with_opsin
from openclatura.fusion import entry_geometry, layout
from openclatura.fusion.faces import select_bounded_face_model, typed_face_model
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _chain(sizes, outgoing_ports=None):
    graph = Chem.RWMol()
    previous = None
    for index, size in enumerate(sizes):
        ring = []
        for local in range(size):
            if previous is not None and local in {1, 2}:
                port = outgoing_ports[index - 1] if outgoing_ports else 3
                ring.append(previous[port + local - 1])
            else:
                atom = Chem.Atom("O" if size == 5 and local == 0 else "C")
                atom.SetIsAromatic(True)
                ring.append(graph.AddAtom(atom))
        for left, right in zip(ring, ring[1:] + ring[:1]):
            if graph.GetBondBetweenAtoms(left, right) is None:
                graph.AddBond(left, right, Chem.BondType.AROMATIC)
        previous = ring
    Chem.SanitizeMol(graph)
    return graph.GetMol()


def _model(graph):
    mol = read_rdkit_mol(graph)
    bounded = select_bounded_face_model(mol, frozenset(mol.atoms))
    assert bounded is not None
    return typed_face_model(mol, bounded)


def _orders(graph):
    original = list(range(graph.GetNumAtoms()))
    shuffled = original[:]
    Random(1604).shuffle(shuffled)
    return original, original[::-1], shuffled


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("count", (4, 8, 16))
def test_deep_chain_exact_roundtrip_in_every_atom_order(count):
    graph = _chain((5,) * count)
    names = set()
    for order in _orders(graph):
        mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
        result = plan_fusion_parent(mol, frozenset(mol.atoms), mode="audited_pin")
        assert isinstance(result, FusionConfirmed), result
        assert result.plan.audit.confirmed
        assert len(result.plan.ast.component_occurrences) == count
        assert isinstance(result.plan.numbering.selected_layout, layout.OpsinEntryLayout)
        assert result.plan.numbering.selected_layout.orientation_score == (0, -count, -count, count, -2 * count)
        name = result.plan.rendered_base_name
        check = verify_with_opsin(name, Chem.MolToSmiles(graph), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        names.add(name)
    assert len(names) == 1


@pytest.mark.parametrize("sizes,ports", (((5,) * 4, None), ((5,) * 6, None), ((5, 6, 6, 5), (3, 3, 3))))
def test_bound_retains_every_exhaustive_winning_perimeter(monkeypatch, sizes, ports):
    graph = _chain(sizes, ports)
    for order in _orders(graph):
        model = _model(Chem.RenumberAtoms(graph, order))
        entry_geometry.entry_direction_geometry.cache_clear()
        bounded = entry_geometry.entry_direction_geometry(model, 25000)
        with monkeypatch.context() as context:
            context.setattr(entry_geometry, "_linear_face_chain", lambda neighbors: False)
            entry_geometry.entry_direction_geometry.cache_clear()
            exhaustive = entry_geometry.entry_direction_geometry(model, 25000)
        entry_geometry.entry_direction_geometry.cache_clear()
        assert bounded is not None
        assert exhaustive is not None
        assert bounded == exhaustive


def test_bent_path_falls_through_to_exhaustive_search(monkeypatch):
    model = _model(_chain((5, 6, 6, 5), (3, 3, 3)))
    entry_geometry.entry_direction_geometry.cache_clear()
    result = entry_geometry.entry_direction_geometry(model, 25000)
    assert result is not None
    assert all(witness.orientation_score[1] > -4 for witness in result[1])
    with monkeypatch.context() as context:
        context.setattr(entry_geometry, "_linear_face_chain", lambda neighbors: False)
        entry_geometry.entry_direction_geometry.cache_clear()
        assert entry_geometry.entry_direction_geometry(model, 25000) == result
    entry_geometry.entry_direction_geometry.cache_clear()


@pytest.mark.parametrize(
    "neighbors",
    (
        {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]},
        {0: [1, 2], 1: [0, 2], 2: [0, 1]},
        {0: [1], 1: [0], 2: [3, 4], 3: [2, 4], 4: [2, 3]},
    ),
)
def test_straight_bound_does_not_admit_nonpaths(neighbors):
    assert not entry_geometry._linear_face_chain(neighbors)


@pytest.mark.parametrize("count", (8, 16))
def test_straight_chain_search_has_a_linear_state_ceiling(monkeypatch, count):
    model = _model(_chain((5,) * count))
    assert layout.preferred_intrinsic_layouts(model)
    budgets = []

    class CountedBudget(layout._Budget):
        def __init__(self, limit):
            super().__init__(limit)
            budgets.append(self)

    entry_geometry.entry_direction_geometry.cache_clear()
    with monkeypatch.context() as context:
        context.setattr(layout, "_Budget", CountedBudget)
        result = entry_geometry.entry_direction_geometry(model, 25000)
    assert result is not None
    assert len(budgets) == 1
    assert budgets[0].used <= 32 * count
    assert len(result[1]) >= 2
    assert all(witness.orientation_score[:2] == (0, -count) for witness in result[1])
    entry_geometry.entry_direction_geometry.cache_clear()


def test_intrinsic_cache_normalizes_defaults_and_preserves_limits(monkeypatch):
    model = _model(_chain((5,) * 4))
    calls = []
    original = layout.intrinsic_fused_layouts

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(layout, "intrinsic_fused_layouts", counted)
        layout._preferred_intrinsic_layouts.cache_clear()
        first = layout.preferred_intrinsic_layouts(model)
        assert first
        assert (
            layout.preferred_intrinsic_layouts(model, search_budget=25000, max_layouts=4096, opsin_ring_map=False)
            is first
        )
        assert len(calls) == 1
        assert layout.preferred_intrinsic_layouts(model, search_budget=25001) == first
        assert len(calls) == 2
        for limits in ({"search_budget": 1}, {"max_layouts": 1}):
            with pytest.raises(layout.LayoutSearchBudgetExceeded):
                layout.preferred_intrinsic_layouts(model, **limits)
        assert len(calls) == 4
        assert layout._preferred_intrinsic_layouts.cache_info().maxsize == 128
    layout._preferred_intrinsic_layouts.cache_clear()
