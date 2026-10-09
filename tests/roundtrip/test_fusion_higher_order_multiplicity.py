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
    # Indazole is an admitted retained component (Table 2.8) and beats
    # pyrazole on ring count, so this graph is no longer named on a pyrazole
    # parent with all three thiophenes under one trithieno prefix. Two now
    # attach to the benzo component and one directly to indazole, and FR-4.9
    # groups identical components by recipient rather than by name, so a
    # group of two is right here. The no-disruption condition of P-25.3.5
    # governs forming a benzoheterocycle by the P-25.2.2.4 procedure; it does
    # not veto a retained component that happens to decompose into benzene
    # plus a heteromonocycle.
    (parent_occurrence,) = plan.ast.parent_occurrences
    parent = next(m for m in plan.ast.component_occurrences if m.occurrence_id == parent_occurrence)
    assert parent.spec_key == "indazole"
    group = next(group for group in plan.ast.multiplicative_groups if len(group.occurrence_ids) == 2)
    by_child = {join.attached_occurrence: join for join in plan.ast.joins}
    for offset, occurrence in enumerate(group.occurrence_ids):
        join = by_child[occurrence]
        assert join.order == 2
        assert {locant.prime_depth for locant in join.attached_locants} == {1 + offset}
        assert {locant.prime_depth for locant in join.host_locants} == {0}

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


# The component arrangement above is settled, but two rendering rules are not
# applied to it yet, so the exact string is still wrong in two independent ways.
# P-25.3.4.2.3.2 keeps a multiplier that is integral to a multipart prefix for
# the alphabetical comparison, so dithieno...benzo sorts before thieno; and
# FR-4.5(c) gives the first-order attached component's locants for fusion to the
# parent priority over minimising its locants towards higher-order components,
# so the benzene attaches at 1,2 rather than 5,6. Both are renderer defects this
# arrangement exposed rather than caused, and neither is a reason to prefer the
# pyrazole parent.
CORRECTED_BASE_NAME = "2H-dithieno[3',4':3,4;3'',4'':5,6]benzo[1,2-g]thieno[3,4-e]indazole"


@pytest.mark.xfail(strict=True, reason="multipart prefix alphabetisation and first-order fusion locants")
def test_the_multipart_prefix_and_first_order_locants_render_correctly(higher_order_case):
    mol = read_rdkit_mol(Chem.Mol(higher_order_case.binary))
    result = plan_fusion_parent(mol, mol.atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    assert result.plan.rendered_base_name == CORRECTED_BASE_NAME
