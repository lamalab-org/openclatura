from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
from rdkit import Chem

from openclatura import name, name_mol
from openclatura.fusion import wrappers
from openclatura.fusion.model import FusionMode
from openclatura.graph_io import read_rdkit_mol, read_smiles
from openclatura.opsin_verify import verify_with_opsin

SMILES = "c1cc2ccc3ccc4ccc5ccc6ccc1c2c3c4c56"
EXPECTED = "13,14-didehydro-9,10-ethanodicyclopenta[c,g]phenanthrene"


def test_bridge_wrapper_proves_completed_system_unsaturation_without_more_search():
    mol = read_smiles(SMILES)
    with patch.object(wrappers, "_systematic_fusion_parent", wraps=wrappers._systematic_fusion_parent) as planner:
        plan = wrappers.plan_bridged_fusion_wrapper(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert planner.call_count == 6
    assert plan is not None and plan.rendered_name == EXPECTED
    assert plan.search_states == 69
    assert plan.parent.name == "dicyclopenta[c,g]phenanthrene"
    assert plan.bridges[0].prefix == "etheno"
    assert plan.bridges[0].internal_bond_orders == (2,)
    assert plan.derivative_state.hydro_operations == ()
    assert plan.saturated_bridge_precursor.prefix == "ethano"
    assert plan.saturated_bridge_precursor.internal_bond_orders == (1,)
    assert len(plan.bridge_unsaturation_operations) == 1
    operation = plan.bridge_unsaturation_operations[0]
    assert operation.locants == ("13", "14")
    assert set(operation.atom_ids) == set(plan.bridges[0].atom_ids)
    assert mol.get_bond(*operation.atom_ids).idx == operation.bond_id
    assert "complete_bridge_pi_assignment" in plan.audit_checks
    assert "".join(part.text for part in plan.rendered_parts) == EXPECTED
    dehydro = next(part for part in plan.rendered_parts if part.grammar_role == "bridge_dehydro")
    assert dehydro.atom_ids == set(operation.atom_ids)
    assert dehydro.bond_ids == {operation.bond_id}


def test_bridge_completed_system_rendering_does_not_require_opsin(monkeypatch):
    def unexpected_opsin():
        pytest.fail("OPSIN must not run during unverified naming")

    monkeypatch.setattr("openclatura.opsin_verify._try_import_py2opsin", unexpected_opsin)
    result = name(SMILES, include_trace=True)

    assert result.name == EXPECTED
    assert result.opsin_check is None
    assert result.parent_nomenclature == "bridged_fusion"
    selected = next(step for step in result.decisions if step.decision == "selected audited bridged fusion parent")
    mol = read_smiles(SMILES)
    plan = wrappers.plan_bridged_fusion_wrapper(mol, mol.atoms, mode=FusionMode.GENERAL)
    operation = plan.bridge_unsaturation_operations[0]
    assert selected.data["bridge_unsaturation_operations"] == [
        {
            "locants": list(operation.locants),
            "atom_ids": list(operation.atom_ids),
            "bond_id": operation.bond_id,
            "bond_order": 2,
        }
    ]
    assert selected.data["saturated_bridge_precursor"]["prefix"] == "ethano"
    assert selected.data["saturated_bridge_precursor"]["internal_bond_orders"] == [1]


@pytest.mark.parametrize("mode", [FusionMode.GENERAL, FusionMode.AUDITED_PIN])
def test_bridge_completed_system_public_name_roundtrips(mode):
    result = name(SMILES, fusion_mode=mode, verify_opsin=True)

    assert result.name == EXPECTED
    assert result.parent_nomenclature == "bridged_fusion"
    assert result.opsin_check is not None and result.opsin_check.status == "matched"


@pytest.mark.parametrize("rotation", [0, 5])
def test_bridge_completed_system_name_is_invariant_to_atom_order(rotation):
    mol = Chem.MolFromSmiles(SMILES)
    order = list(reversed(range(mol.GetNumAtoms())))
    order = order[rotation:] + order[:rotation]
    result = name_mol(Chem.RenumberAtoms(mol, order), verify_opsin=True)

    assert result.name == EXPECTED
    assert result.opsin_check is not None and result.opsin_check.status == "matched"


@pytest.mark.parametrize(
    "field,value",
    [("locants", ("9", "10")), ("atom_ids", (0, 1)), ("bond_order", 1), ("bond_id", -1), ("empty", None)],
)
def test_bridge_dehydrogenation_audit_rejects_corrupted_operations(field, value):
    mol = read_smiles(SMILES)
    plan = wrappers.plan_bridged_fusion_wrapper(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert plan is not None
    corrupted = () if field == "empty" else (replace(plan.bridge_unsaturation_operations[0], **{field: value}),)

    assert (
        wrappers._audit_bridge_plan(
            mol,
            frozenset(mol.atoms),
            plan.parent.atom_ids,
            dict(plan.parent.locant_maps[0]),
            plan.bridges,
            plan.parent.selected_bond_model,
            plan.derivative_state,
            corrupted,
            plan.saturated_bridge_precursor,
        )
        is None
    )


def test_bridge_dehydrogenation_requires_its_saturated_precursor():
    mol = read_smiles(SMILES)
    plan = wrappers.plan_bridged_fusion_wrapper(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert plan is not None
    with pytest.raises(ValueError, match="saturated precursor"):
        replace(plan, bridge_unsaturation_operations=())
    with pytest.raises(ValueError, match="saturated precursor"):
        replace(plan, saturated_bridge_precursor=None)
    assert (
        wrappers._audit_bridge_plan(
            mol,
            frozenset(mol.atoms),
            plan.parent.atom_ids,
            dict(plan.parent.locant_maps[0]),
            plan.bridges,
            plan.parent.selected_bond_model,
            plan.derivative_state,
            plan.bridge_unsaturation_operations,
            replace(plan.saturated_bridge_precursor, internal_bond_orders=(2,)),
        )
        is None
    )


def test_graph_built_longer_conjugated_bridge_uses_four_dehydro_locants():
    original = read_smiles(SMILES)
    plan = wrappers.plan_bridged_fusion_wrapper(original, original.atoms, mode=FusionMode.GENERAL)
    assert plan is not None
    left, right = plan.bridges[0].atom_ids
    graph = Chem.RWMol(Chem.MolFromSmiles(SMILES))
    graph.RemoveBond(left, right)
    middle = [graph.AddAtom(Chem.Atom("C")) for _ in range(2)]
    for atom in middle:
        graph.GetAtomWithIdx(atom).SetIsAromatic(True)
    path = (left, *middle, right)
    for first, second in zip(path, path[1:]):
        graph.AddBond(first, second, Chem.BondType.AROMATIC)
    Chem.SanitizeMol(graph)
    mol = read_rdkit_mol(graph)
    # Extend the parent's proved Kekule assignment with a conjugated path;
    # RDKit may otherwise choose a resonance form double-bonded to a bridgehead.
    for edge, order in plan.derivative_state.bond_delta.assignment.orders:
        mol.set_bond_order(mol.get_bond(*edge).idx, order)
    for atom, endpoint in zip((left, right), plan.bridges[0].endpoint_atom_ids):
        mol.set_bond_order(mol.get_bond(atom, endpoint).idx, 1)
    for index, (first, second) in enumerate(zip(path, path[1:])):
        mol.set_bond_order(mol.get_bond(first, second).idx, 2 if index % 2 == 0 else 1)
    locants = dict(plan.parent.locant_maps[0])
    bridges = wrappers._bridge_operations(mol, (path,), plan.parent.atom_ids, locants)
    assert bridges is not None
    unsaturation = wrappers._bridge_dehydrogenation(mol, bridges, locants)
    assert unsaturation is not None
    assert [locant for operation in unsaturation for locant in operation.locants] == ["13", "14", "15", "16"]
    precursor = wrappers._saturated_bridge_precursor(bridges[0])
    derivative = wrappers.parent_derivative_state(
        mol, plan.parent.atom_ids, plan.parent.selected_bond_model, locants, preserve_retained_parent_state=True
    )
    checks = wrappers._audit_bridge_plan(
        mol,
        frozenset(mol.atoms),
        plan.parent.atom_ids,
        locants,
        bridges,
        plan.parent.selected_bond_model,
        derivative,
        unsaturation,
        precursor,
    )
    assert checks is not None and "complete_bridge_pi_assignment" in checks
    rendered = "".join(part.text for part in wrappers._render_bridge_parts(mol, plan.parent, bridges, unsaturation))
    assert rendered == "13,14,15,16-tetradehydro-9,10-butanodicyclopenta[c,g]phenanthrene"
    assert verify_with_opsin(rendered, Chem.MolToSmiles(graph)).status == "matched"
