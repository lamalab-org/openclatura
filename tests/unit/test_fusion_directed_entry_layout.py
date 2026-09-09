"""Directed component entries preserve complete fused numbering and graph state."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion import layout
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol
from openclatura.opsin_verify import verify_with_opsin

SEED = "CCCN(CCC)CC(=O)N1c2cc(F)ccc2-n2c(n[nH]c2=O)-c2cccnc21"


def _graph(variant, ordering=0):
    graph = Chem.RWMol(Chem.MolFromSmiles(SEED))
    if variant == "chloro":
        graph.GetAtomWithIdx(14).SetAtomicNum(17)
    elif variant == "extra_aza":
        graph.GetAtomWithIdx(12).SetAtomicNum(7)
    elif variant == "diaza_neighbor":
        graph.GetAtomWithIdx(12).SetAtomicNum(7)
        graph.GetAtomWithIdx(15).SetAtomicNum(7)
    elif variant == "carbon_neighbor":
        graph.GetAtomWithIdx(28).SetAtomicNum(6)
    elif variant == "methyl":
        carbon = graph.AddAtom(Chem.Atom(6))
        graph.AddBond(25, carbon, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    order = list(range(graph.GetNumAtoms()))
    if ordering == 1:
        order.reverse()
    elif ordering == 2:
        random.Random(53).shuffle(order)
    return Chem.RenumberAtoms(graph, order)


def _context(graph):
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed), result
    registry = fusion_component_registry()
    ast = result.plan.ast
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in ast.component_occurrences}
    from openclatura.fusion.faces import cached_bounded_face_model, typed_face_model

    model = typed_face_model(mol, cached_bounded_face_model(mol, atoms))
    return mol, model, ast, specs


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("variant", ("original", "chloro", "extra_aza", "diaza_neighbor", "carbon_neighbor", "methyl"))
@pytest.mark.parametrize("ordering", range(3))
def test_directed_entry_congeners_roundtrip_without_core_name_lookups(variant, ordering):
    graph = _graph(variant, ordering)
    result = name_mol(graph, fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None
    assert "cyclo[" not in result.name
    check = verify_with_opsin(result.name, Chem.MolToSmiles(graph), standardize_smiles=False)
    assert check.status == "matched", (result.name, check)


def test_directed_layout_has_a_complete_audited_three_ring_row():
    mol, model, ast, specs = _context(_graph("original"))
    layouts = layout.component_entry_layouts(model, ast, specs)
    assert len(layouts) == 1
    witness = layouts[0]
    assert witness.component_entry_edge is not None
    assert mol.get_bond(*witness.component_entry_edge) is not None
    assert witness.orientation_score[1] == -3
    assert {atom for atom, _, _ in witness.atom_positions} == {atom for face in model.faces for atom in face.atom_cycle}
    assert layout._audit_layout(
        model,
        {face.id: face.atom_cycle for face in model.faces},
        {atom: (x, y) for atom, x, y in witness.atom_positions},
    )


def test_directed_layout_obeys_the_existing_search_budget():
    _, model, ast, specs = _context(_graph("original"))
    with pytest.raises(layout.LayoutSearchBudgetExceeded):
        layout.component_entry_layouts(model, ast, specs, search_budget=1)


@pytest.mark.parametrize("variant", ("extra_aza", "carbon_neighbor"))
def test_multiplied_components_preserve_their_directed_interface(variant):
    _, model, ast, specs = _context(_graph(variant))
    assert ast.multiplicative_groups
    layouts = layout.component_entry_layouts(model, ast, specs)
    assert layouts
    assert layouts[0].component_entry_edge in {join.interface.ordered_input_atoms for join in ast.joins}


def test_corrupted_directed_geometry_is_rejected(monkeypatch):
    _, model, ast, specs = _context(_graph("original"))
    monkeypatch.setattr(layout, "_audit_layout", lambda *_: False)
    assert layout.component_entry_layouts(model, ast, specs) == ()


def test_invalid_entry_metadata_cannot_escape_the_layout():
    _, model, ast, specs = _context(_graph("original"))
    witness = layout.component_entry_layouts(model, ast, specs)[0]
    with pytest.raises(ValueError, match="component entry"):
        replace(witness, component_entry_edge=(-1, -1))


@pytest.mark.parametrize("missing", (False, True))
def test_final_numbering_audit_rejects_changed_entry_provenance(missing):
    from openclatura.fusion.audit import _audit_layout_numbering_compatibility

    mol, model, _, _ = _context(_graph("original"))
    atoms = frozenset(atom for face in model.faces for atom in face.atom_cycle)
    plan = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN).plan
    witness = plan.numbering.selected_layout
    changed = replace(witness, component_entry_edge=None if missing else tuple(reversed(witness.component_entry_edge)))
    proof = replace(plan.numbering, selected_layout=changed)
    errors = []
    _audit_layout_numbering_compatibility(atoms, proof, errors, ast=plan.ast)
    assert any("entry" in error for error in errors)


def test_directed_plan_does_not_pay_for_unused_intrinsic_search(monkeypatch):
    from openclatura.fusion import planner

    mol = read_rdkit_mol(_graph("original"))
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms

    def forbidden(*args, **kwargs):
        raise AssertionError("directed plan must not enumerate discarded intrinsic layouts")

    monkeypatch.setattr(planner, "preferred_intrinsic_layouts", forbidden)
    assert isinstance(plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN), FusionConfirmed)


def test_failed_directed_audit_does_not_restart_unrestricted_numbering(monkeypatch):
    from openclatura.fusion import planner

    _, model, _, _ = _context(_graph("original"))
    mol = read_rdkit_mol(_graph("original"))
    monkeypatch.setattr(planner, "component_entry_layouts", lambda *_: ())

    def forbidden(*args, **kwargs):
        raise AssertionError("numbering must not run after the directed witness failed")

    monkeypatch.setattr(planner, "completed_system_numbering_selection", forbidden)
    atoms = frozenset(atom for face in model.faces for atom in face.atom_cycle)
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert not isinstance(result, FusionConfirmed)


@pytest.mark.parametrize(
    "changes",
    (
        {"opposite_ports": (0, 0)},
        {"opposite_ports": (0, 1)},
        {"directed_entry_port": 99},
        {"directed_entry_port": None},
        {"entry_component_size": 0},
    ),
)
def test_invalid_direction_policy_is_not_accepted(changes):
    shape = next(shape for shape in layout.RING_SHAPE_TEMPLATES if shape.directed_entry_port is not None)
    with pytest.raises(ValueError):
        replace(shape, **changes)
