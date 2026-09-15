"""Higher-order citation grammar and graph round trips share component scopes."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion import descriptor
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import FusionCitationNode, FusionConfirmed, FusionDescriptor
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol
from roundtrip.random_fusion_helpers import random_fusion_cases


@pytest.fixture
def higher_order_case():
    return random_fusion_cases(count=2)[1]


def test_multiplicative_depth_is_added_to_higher_order_depth(higher_order_case):
    mol = read_rdkit_mol(Chem.Mol(higher_order_case.binary))
    result = plan_fusion_parent(mol, mol.atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    group = next(group for group in plan.ast.multiplicative_groups if len(group.occurrence_ids) == 3)
    by_child = {join.attached_occurrence: join for join in plan.ast.joins}
    for offset, occurrence in enumerate(group.occurrence_ids):
        join = by_child[occurrence]
        assert join.order == 2
        assert {locant.prime_depth for locant in join.attached_locants} == {1 + offset}
        assert {locant.prime_depth for locant in join.host_locants} == {0}
    assert "[3',4':1,2;3'',4'':3,4;3''',4''':5,6]" in plan.rendered_base_name

    target = by_child[group.occurrence_ids[0]]
    damaged = replace(
        target,
        interface=replace(
            target.interface,
            attached_path=tuple(replace(locant, prime_depth=0) for locant in target.interface.attached_path),
            cited_attached_locants=tuple(
                replace(locant, prime_depth=0) for locant in target.interface.cited_attached_locants
            ),
        ),
    )
    joins = tuple(damaged if join is target else join for join in plan.ast.joins)
    ast = replace(plan.ast, joins=joins, descriptors=tuple(FusionDescriptor.from_interface(j.interface) for j in joins))
    audit = audit_fusion_plan(
        mol,
        mol.atoms,
        ast=ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        derivative_state=plan.derivative_state,
        indicated_hydrogens=plan.indicated_hydrogens,
        registry=fusion_component_registry(),
    )
    assert not audit.confirmed
    assert any("prime depth" in error for error in audit.errors)


@pytest.mark.opsin
def test_higher_order_multiplied_components_roundtrip_in_both_atom_orders(higher_order_case):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    mol = Chem.Mol(higher_order_case.binary)
    names = []
    for graph in (mol, Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))):
        result = name_mol(graph, include_trace=True)
        assert result.parent_nomenclature == "systematic_fusion"
        assert verify_with_opsin(result.name, higher_order_case.smiles, standardize_smiles=False).status == "matched"
        names.append(result.name)
    assert names[0] == names[1]


def test_scope_check_distinguishes_sibling_return_from_nested_descent():
    leaf = FusionCitationNode(2)
    branch = FusionCitationNode(3, (FusionCitationNode(4),))
    invalid = FusionCitationNode(0, (FusionCitationNode(1, (leaf, branch)),))
    valid = FusionCitationNode(0, (FusionCitationNode(1, (branch, leaf)),))
    assert not descriptor._tree_citation_scope_supported(invalid, ())
    assert descriptor._tree_citation_scope_supported(valid, ())
    assert descriptor._tree_citation_scope_supported(FusionCitationNode(0, (leaf, branch)), ())
