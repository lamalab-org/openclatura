from __future__ import annotations

import pytest

from openclatura.fusion.model import FusionMode
from openclatura.fusion.wrappers import (
    NondetachableBridgeKind,
    WrapperParentKind,
    plan_bridged_fusion_wrapper,
    plan_fusion_spiro_side,
)
from openclatura.graph_io import read_smiles
from openclatura.namer import _spiro_subgraph_assembly


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C12=CC=C(C3=CC=CC=C13)O2", "1,4-epoxynaphthalene"),
        ("C12=CC=C(C=3C4=CC=C(C13)C4)O2", "1,4-epoxy-5,8-methanonaphthalene"),
        (
            "C12=CC(=CC3=CC=4C5=CC=C(C4C=C13)O5)C2",
            "5,8-epoxy-1,3-methanoanthracene",
        ),
        ("N12C=CC3=CC(=CC=C13)C2", "1,5-methanoindole"),
    ],
)
def test_bridged_fusion_wrapper_uses_retained_parent_locants(smiles, expected):
    mol = read_smiles(smiles)

    plan = plan_bridged_fusion_wrapper(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert plan is not None
    assert plan.rendered_name == expected
    assert plan.parent.kind is WrapperParentKind.RETAINED
    assert plan.audit_ok
    assert "".join(part.text for part in plan.rendered_parts) == expected
    assert {atom for bridge in plan.bridges for atom in bridge.atom_ids} | set(plan.parent.atom_ids) == set(mol.atoms)


def test_bridged_fusion_wrapper_records_bridge_roles_and_local_graph_ownership():
    mol = read_smiles("C12=CC=C(C=3C4=CC=C(C13)C4)O2")

    plan = plan_bridged_fusion_wrapper(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert plan is not None
    assert {bridge.kind for bridge in plan.bridges} == {
        NondetachableBridgeKind.CARBO,
        NondetachableBridgeKind.EPOXY,
    }
    for bridge in plan.bridges:
        assert len(bridge.endpoint_atom_ids) == 2
        assert len(bridge.endpoint_locants) == 2
        assert bridge.bond_ids
    bridge_tokens = [part for part in plan.rendered_parts if part.grammar_role == "nondetachable_bridge"]
    assert {part.text.rstrip("-") for part in bridge_tokens} == {"epoxy", "methano"}
    assert all(part.source == "fusion_wrapper_renderer" for part in plan.rendered_parts)


def test_ordinary_fused_parent_is_not_misclassified_as_a_bridge_wrapper():
    mol = read_smiles("O1C2=C(C=C1)C=CS2")

    assert plan_bridged_fusion_wrapper(mol, mol.atoms, mode=FusionMode.GENERAL) is None


def test_fusion_spiro_side_consumes_the_proof_locant_without_name_parsing():
    mol = read_smiles("O1C2=C(C=C1)C=CS2")

    side = plan_fusion_spiro_side(mol, mol.atoms, 4, mode=FusionMode.GENERAL)

    assert side is not None
    assert side.parent.kind is WrapperParentKind.SYSTEMATIC_FUSION
    assert side.parent.name == "thieno[2,3-b]furan"
    assert side.junction_locant == dict(side.parent.locant_maps[0])[4] == "2"
    assembly = side.to_spiro_assembly(parent_locant="7")
    assert assembly.parent_locant == "7"
    assert assembly.side_locant == "2"
    assert assembly.side_parent_name == "thieno[2,3-b]furan"


def test_spiro_subgraph_prefers_typed_fusion_side_in_general_mode(monkeypatch):
    mol = read_smiles("O1C2=C(C=C1)C=CS2")
    monkeypatch.setattr("openclatura.fusion.context.current_fusion_mode", lambda: FusionMode.GENERAL)

    assembly = _spiro_subgraph_assembly(mol, 4, set(mol.atoms))

    assert assembly.side_parent_name == "thieno[2,3-b]furan"
    assert assembly.side_locant == "2"
    assert assembly.side_prefixes == ()

