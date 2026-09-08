from copy import deepcopy
from unittest.mock import patch

import pytest

from openclatura.fusion import faces as face_search
from openclatura.fusion.config import fusion_nomenclature_config, fusion_nomenclature_config_from_data
from openclatura.fusion.descriptor import render_fusion_name_parts
from openclatura.fusion.layout import LayoutSearchBudgetExceeded, intrinsic_fused_layouts
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.numbering import completed_system_numberings
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_smiles
from openclatura.molecule import Molecule
from openclatura.naming_data import load_json_table


def _bicycle(n, m, ids=None):
    ids = list(range(n + m - 2)) if ids is None else ids
    cycles = (tuple(ids[:n]), (ids[0], ids[1], *ids[n:]))
    edges = {tuple(sorted(edge)) for cycle in cycles for edge in zip(cycle, cycle[1:] + cycle[:1])}
    mol = Molecule()
    for atom in ids:
        mol.add_atom("C", idx=atom)
    for idx, (left, right) in enumerate(sorted(edges)):
        mol.add_bond(left, right, idx=idx)
    return mol


PAIRS = [(large, small) for large in range(9, 21) for small in range(5, large + 1)]


@pytest.mark.parametrize(("large", "small"), PAIRS)
def test_two_path_proof_bypasses_exponential_cycle_search(monkeypatch, large, small):
    def forbidden(*args, **kwargs):
        pytest.fail("ordinary carbon bicycles must not enumerate cycles or face subsets")

    monkeypatch.setattr(face_search, "enumerate_chordless_cycles", forbidden)
    monkeypatch.setattr(face_search, "_independent_face_sets", forbidden)
    mol = _bicycle(large, small)
    bounded = face_search.select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None and bounded.audit.ok
    assert sorted(len(face.atoms) for face in bounded.faces) == sorted((large, small))
    assert bounded.cycle_rank == 2
    assert not bounded.interior_atoms
    typed = face_search.typed_face_model(mol, bounded)
    layouts = intrinsic_fused_layouts(typed)
    assert len(layouts) == 4
    assert all(layout.orientation_score[1] == -2 for layout in layouts)
    numberings = completed_system_numberings(mol, bounded, face_model=typed, layouts=layouts)
    assert numberings
    assert all(
        {str(locant) for _, locant in numbering.atom_to_locant if locant.fusion_suffix}
        == {f"{small - 2}a", f"{large + small - 4}a"}
        for numbering in numberings
    )


def test_large_bicycle_maps_are_invariant_under_graph_atom_relabelling():
    size = 12
    ids = [83, 19, 62, 7, 105, 34, 91, 56, 2, 44, 71, 28]
    maps = []
    for labels in (list(range(size)), ids):
        mol = _bicycle(9, 5, labels)
        bounded = face_search.select_bounded_face_model(mol, mol.atoms)
        assert bounded is not None
        maps.append(
            {
                tuple(numbering.string_map[atom] for atom in labels)
                for numbering in completed_system_numberings(mol, bounded)
            }
        )
    assert maps[0] == maps[1]


@pytest.mark.parametrize("kind", ["oversized", "hetero", "charged", "bridged", "spiro", "third_face"])
def test_large_carbon_extension_rejects_other_topologies(kind):
    mol = _bicycle(21 if kind == "oversized" else 9, 5)
    if kind == "hetero":
        mol = read_smiles("N1CCCCCCC2CCCC12")
    elif kind == "charged":
        mol = read_smiles("[CH-]1CCCCCCC2CCCC12")
    elif kind == "bridged":
        mol = read_smiles("C1CCCC2CCCCCC1CC2")
    elif kind == "spiro":
        mol = read_smiles("C1CCCCCCC2(C1)CCCC2")
    elif kind == "third_face":
        mol = read_smiles("C1CCCCCCC2CCC3CCCC3C12")
    assert face_search.select_bounded_face_model(mol, mol.atoms) is None


def test_global_cycle_and_shape_bounds_remain_unchanged():
    config = fusion_nomenclature_config()
    assert config.search.maximum_ring_size == 8
    assert config.annulene_ring_sizes == tuple(range(7, 21))
    assert max(shape.ring_size for shape in config.ring_shapes) == 8
    with pytest.raises(ValueError, match="max_size"):
        face_search.enumerate_chordless_cycles(_bicycle(9, 5), max_size=9)


def test_explicit_search_override_does_not_reuse_default_fast_path_cache():
    mol = _bicycle(9, 5)
    assert face_search.cached_bounded_face_model(mol, mol.atoms) is not None
    assert face_search.cached_bounded_face_model(mol, mol.atoms, max_ring_size=8) is None
    assert face_search.cached_bounded_face_model(mol, mol.atoms, max_ring_size=7) is None
    assert face_search.cached_bounded_face_model(mol, mol.atoms, min_ring_size=5) is None
    assert face_search.cached_bounded_face_model(mol, mol.atoms) is not None


@pytest.mark.parametrize("field", ["cycle_search_budget", "model_search_budget"])
def test_fast_path_rejects_invalid_search_budgets(field):
    mol = _bicycle(9, 5)
    with pytest.raises(ValueError, match="budgets must be positive"):
        face_search.select_bounded_face_model(mol, mol.atoms, **{field: 0})


def test_disabling_annulene_policy_disables_fast_path(monkeypatch):
    data = deepcopy(load_json_table("fusion_components.json"))
    del data["annulene_series"]
    config = fusion_nomenclature_config_from_data(data)
    assert config.annulene_ring_sizes == ()
    monkeypatch.setattr(face_search, "_CONFIG", config)
    mol = _bicycle(9, 5)
    assert face_search.select_bounded_face_model(mol, mol.atoms) is None


@pytest.mark.parametrize(("budget", "limit"), [(3, 4), (4, 3)])
def test_two_ring_layout_adapter_honors_budgets(budget, limit):
    mol = _bicycle(9, 5)
    bounded = face_search.select_bounded_face_model(mol, mol.atoms)
    typed = face_search.typed_face_model(mol, bounded)
    with pytest.raises(LayoutSearchBudgetExceeded):
        intrinsic_fused_layouts(typed, search_budget=budget, max_layouts=limit)


@pytest.mark.parametrize("maximum", [6, True, 0])
def test_annulene_policy_rejects_invalid_intervals(maximum):
    data = deepcopy(load_json_table("fusion_components.json"))
    data["annulene_series"]["maximum_ring_size"] = maximum
    with pytest.raises(ValueError):
        fusion_nomenclature_config_from_data(data)


def test_generated_annulenes_reuse_graph_templates_and_renderer_bindings():
    registry = fusion_component_registry()
    for size in (7, 8):
        components = [
            item
            for item in registry.components
            if len(item.spec.atoms) == size
            and len(item.spec.rings) == 1
            and all(atom.symbol == "C" for atom in item.spec.atoms)
        ]
        assert len(components) == 1
        assert components[0].spec.parent_name == f"[{size}]annulene"
        assert components[0].spec.usable_as_parent and components[0].spec.usable_as_attached
    for size in range(9, 21):
        spec = registry.get(f"[{size}]annulene").spec
        assert len(spec.atoms) == len(spec.bonds) == size
        assert len(spec.rings) == 1
        assert spec.template.family == "generated_monocycle"
    mol = read_smiles("C1CCCCCCC2CCCC12")
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    parts = render_fusion_name_parts(result.plan.ast, registry, mol=mol)
    assert "".join(part.text for part in parts) == "cyclopenta[9]annulene"
    assert [part.grammar_role for part in parts] == ["fusion_attached_component", "fusion_parent_component"]
    assert all(part.source == "fusion_renderer" and part.atom_ids and part.bond_ids for part in parts)
    assert set().union(*(set(part.atom_ids) for part in parts)) == set(mol.atoms)
    assert set().union(*(set(part.bond_ids) for part in parts)) == set(mol.bonds)
    assert len(result.plan.ast.joins) == 1
    assert len(result.plan.ast.descriptors) == 1


@pytest.mark.parametrize("sizes", [(7, 5), (8, 5), (8, 6)])
def test_smaller_annulenes_reuse_existing_cycle_and_shape_search(sizes, monkeypatch):
    mol = _bicycle(*sizes)
    with patch.object(
        face_search, "enumerate_chordless_cycles", wraps=face_search.enumerate_chordless_cycles
    ) as search:
        bounded = face_search.select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None
    search.assert_called_once()

    def forbidden(*args, **kwargs):
        pytest.fail("7/8-member rings already have ordinary shape templates")

    monkeypatch.setattr("openclatura.fusion.layout._ordinary_large_bicycle_layouts", forbidden)
    assert intrinsic_fused_layouts(face_search.typed_face_model(mol, bounded))
