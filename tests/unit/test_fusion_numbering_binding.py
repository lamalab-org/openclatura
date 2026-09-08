from dataclasses import replace

import pytest

from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import FusionConfirmed, FusionMode, PinDecision, PinStatus
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.wrappers import _systematic_fusion_parent
from openclatura.graph_io import read_smiles
from openclatura.naming_context import NamingIntent
from openclatura.parent_pipeline import build_parent_assembly_plan
from openclatura.parent_selection import ParentSelection
from openclatura.ring_parent import RingParent


def _plans(smiles):
    mol = read_smiles(smiles)
    atoms = {atom for atom in mol.atoms if len(mol.get_neighbors(atom)) > 1}
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    alternatives = result.plan.numbering_variants
    assert len(alternatives) == 2
    return mol, atoms, result.plan, alternatives


def _audit(mol, atoms, plan, **changes):
    arguments = dict(
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        derivative_state=plan.derivative_state,
        charge_operations=plan.charge_operations,
        lambda_descriptors=plan.lambda_descriptors,
    )
    arguments.update(changes)
    return audit_fusion_plan(mol, atoms, **arguments)


def test_saturated_ch2_sites_do_not_break_topologically_tied_maps():
    mol, atoms, plan, alternatives = _plans("CC1CCCCCC2CCCC2C1")
    site = next(atom for atom in atoms if any(neighbor not in atoms for neighbor in mol.get_neighbors(atom)))
    assert plan.indicated_hydrogens == ()
    assert {numbered.numbering.string_input_locant_maps()[0][site] for numbered in alternatives} == {"5", "9"}
    assert all(_audit(mol, atoms, numbered).confirmed for numbered in alternatives)


def test_assembly_selects_the_whole_numbered_plan_without_replanning(monkeypatch):
    mol, atoms, plan, alternatives = _plans("OC1=CC2CCCC2CCCCCC1")
    site = next(atom for atom in atoms if any(neighbor not in atoms for neighbor in mol.get_neighbors(atom)))
    hydride = RingParent.from_fusion_plan(plan, pin_decision=PinDecision(PinStatus.VALID_GENERAL_NAME, ()))
    selection = ParentSelection([list(atoms)], True, True, False, False, (8, 3, 0), ring_parent=hydride)
    mol._fusion_plan_cache.clear()

    def forbidden(*args, **kwargs):
        pytest.fail("assembly must reuse the first-class audited numbering alternatives")

    monkeypatch.setattr("openclatura.fusion.planner._complete_fusion_plan", forbidden)
    assembly = build_parent_assembly_plan(mol, selection, NamingIntent.component((site,)), {}, parent_hydride=hydride)
    selected = assembly.parts.parent_hydride.fusion_plan
    assert any(selected is alternative for alternative in alternatives)
    assert assembly.locant_map[site] == "5"
    assert selected.numbering.string_input_locant_maps()[0] == assembly.locant_map
    assert _audit(mol, atoms, selected).confirmed
    assert sum(len(operation.locants) for operation in selected.derivative_state.hydro_operations) == 10
    assert assembly.parts.parent_hydride.hydride_metadata.default_indicated_h == ("1",)
    assert {str(locant) for locant in selected.indicated_hydrogens} == {"1"}


@pytest.mark.parametrize("field", ["bond_model", "numbering", "derivative_state"])
def test_partial_ten_plus_five_audit_rejects_cross_numbering_mutations(field):
    mol, atoms, _, alternatives = _plans("OC1=CC2CCCC2CCCCCC1")
    left, right = alternatives
    assert getattr(left, field) != getattr(right, field)
    assert _audit(mol, atoms, left).confirmed
    assert not _audit(mol, atoms, left, **{field: getattr(right, field)}).confirmed


def test_numbering_alternatives_survive_molecule_cache_clear_and_plan_copy():
    mol, _, plan, alternatives = _plans("CC1CCCCCC2CCCC2C1")
    mol._fusion_plan_cache.clear()
    assert replace(plan).numbering_variants == alternatives
    assert all(not variant.numbering_variants for variant in alternatives)


def test_numbering_variants_cannot_drop_maps_or_nest_aggregate_plans():
    _, _, plan, alternatives = _plans("CC1CCCCCC2CCCC2C1")
    with pytest.raises(ValueError, match="every and only"):
        replace(plan, numbering_variants=alternatives[:1])
    with pytest.raises(ValueError, match="single-map leaves"):
        replace(plan, numbering_variants=(plan,))


def test_wrapper_handoff_preserves_each_numberings_model_and_leaf_plan():
    mol, atoms, _, alternatives = _plans("OC1=CC2CCCC2CCCCCC1")
    wrapped = _systematic_fusion_parent(mol, frozenset(atoms), FusionMode.AUDITED_PIN)
    assert wrapped is not None
    assert wrapped.bond_models == tuple(plan.bond_model for plan in alternatives)
    assert wrapped.bond_models[0] != wrapped.bond_models[1]
    mol._fusion_plan_cache.clear()
    for index, variant in enumerate(alternatives):
        selected = wrapped.select(index)
        assert selected.fusion_plan is variant
        assert selected.selected_bond_model is variant.bond_model
        assert selected.hydride.bond_model is variant.bond_model
        assert selected.locant_maps == (tuple(sorted(variant.numbering.string_input_locant_maps()[0].items())),)
        assert selected.hydride.hydride_metadata.default_indicated_h == tuple(map(str, variant.indicated_hydrogens))
        assert selected.fusion_plan.derivative_state is variant.derivative_state
        assert _audit(mol, atoms, selected.fusion_plan).confirmed
        assert selected.select(0) == selected


def test_wrapper_selection_rejects_a_bond_model_borrowed_from_another_numbering():
    mol, atoms, _, _ = _plans("OC1=CC2CCCC2CCCCCC1")
    wrapped = _systematic_fusion_parent(mol, frozenset(atoms), FusionMode.AUDITED_PIN)
    corrupted = replace(wrapped, bond_models=(wrapped.bond_models[0], wrapped.bond_models[0]))
    with pytest.raises(ValueError, match="aligned fusion proof"):
        corrupted.select(1)
