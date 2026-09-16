"""Charged sigma-donor parents select a realizable fusion pi citation."""

from dataclasses import replace
from random import Random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion import planner
from openclatura.fusion.citation_pi_donor import charged_donor_citation_pi_dead_end
from openclatura.fusion.model import FusionConfirmed
from openclatura.graph_io import read_rdkit_mol


def _graph(donor_ligand="O", substituent=None):
    graph = Chem.RWMol()
    for index in range(17):
        atom = Chem.Atom("N" if index in {1, 8, 9, 16} else "C")
        if index == 9:
            atom.SetFormalCharge(1)
        graph.AddAtom(atom)
    doubles = {(0, 8), (1, 2), (3, 4), (5, 9), (6, 7), (10, 11), (12, 13), (14, 15)}
    for edge in (
        (0, 1),
        (0, 8),
        (1, 2),
        (2, 3),
        (2, 7),
        (3, 4),
        (4, 5),
        (4, 16),
        (5, 6),
        (5, 9),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (10, 15),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
    ):
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in doubles else Chem.BondType.SINGLE)
    oxide = Chem.Atom("O")
    oxide.SetFormalCharge(-1)
    graph.AddBond(9, graph.AddAtom(oxide), Chem.BondType.SINGLE)
    graph.AddBond(16, graph.AddAtom(Chem.Atom(donor_ligand)), Chem.BondType.SINGLE)
    if substituent:
        graph.AddBond(0, graph.AddAtom(Chem.Atom(substituent)), Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


def _orders(graph):
    order = list(range(graph.GetNumAtoms()))
    shuffled = order[:]
    Random(13086).shuffle(shuffled)
    return order, order[::-1], shuffled


def _plan(graph):
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = planner.plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    assert result.plan.audit.confirmed
    return mol, result.plan


def test_rejected_orientation_has_no_completion_for_surviving_child_edge(monkeypatch):
    source = _graph()
    for order in _orders(source):
        with monkeypatch.context() as context:
            context.setattr(planner, "charged_donor_citation_pi_dead_end", lambda mol, plan: None)
            mol, plan = _plan(Chem.RenumberAtoms(source, order))
        plan = next(variant for variant in plan.numbering_variants if charged_donor_citation_pi_dead_end(mol, variant))
        edge = charged_donor_citation_pi_dead_end(mol, plan)
        assert edge is not None
        locants = plan.numbering.string_input_locant_maps()[0]
        assert {locants[atom] for atom in edge} == {"1", "2"}
        assert all(dict(assignment.orders)[edge] == 1 for assignment in plan.bond_model.allowed_kekule_assignments)
        assert len(plan.bond_model.allowed_kekule_assignments) == 2
        assert all(
            sum(value == 2 for _, value in assignment.orders) == 8
            for assignment in plan.bond_model.allowed_kekule_assignments
        )
        assert not plan.derivative_state.hydro_operations
        assert not plan.derivative_state.unsaturation_operations
        assert not plan.derivative_state.added_hydrogen_operations


@pytest.mark.parametrize("missing_proof", ("charge", "graph_charge", "assignments", "required_double"))
def test_incomplete_or_out_of_domain_proofs_do_not_reject(monkeypatch, missing_proof):
    with monkeypatch.context() as context:
        context.setattr(planner, "charged_donor_citation_pi_dead_end", lambda mol, plan: None)
        mol, plan = _plan(_graph())
    plan = next(variant for variant in plan.numbering_variants if charged_donor_citation_pi_dead_end(mol, variant))
    edge = charged_donor_citation_pi_dead_end(mol, plan)
    assert edge is not None
    if missing_proof == "charge":
        plan = replace(plan, charge_operations=())
    elif missing_proof == "graph_charge":
        graph = plan.abstract_parent_graph
        plan = replace(
            plan,
            abstract_parent_graph=replace(graph, atoms=(replace(graph.atoms[0], formal_charge=1), *graph.atoms[1:])),
        )
    else:
        model = plan.bond_model
        double = next(edge for edge, order in model.allowed_kekule_assignments[0].orders if order == 2)
        model = replace(
            model,
            **(
                {"allowed_kekule_assignments": ()}
                if missing_proof == "assignments"
                else {
                    "required_double_bonds": frozenset({double}),
                    "pi_eligible_edges": model.pi_eligible_edges - {double},
                }
            ),
        )
        plan = replace(plan, bond_model=model)
    assert charged_donor_citation_pi_dead_end(mol, plan) is None


@pytest.mark.parametrize("unproved_role", ("neutral_oxide_ligand", "unsubstituted_donor"))
def test_parent_matching_alone_does_not_prove_parser_valence(monkeypatch, unproved_role):
    source = _graph()
    with monkeypatch.context() as context:
        context.setattr(planner, "charged_donor_citation_pi_dead_end", lambda mol, plan: None)
        mol, plan = _plan(source)
    plan = next(variant for variant in plan.numbering_variants if charged_donor_citation_pi_dead_end(mol, variant))
    changed = Chem.RWMol(source)
    if unproved_role == "neutral_oxide_ligand":
        changed.GetAtomWithIdx(17).SetFormalCharge(0)
    else:
        changed.RemoveAtom(18)
        changed.GetAtomWithIdx(16).SetNumExplicitHs(1)
    Chem.SanitizeMol(changed)
    assert charged_donor_citation_pi_dead_end(read_rdkit_mol(changed), plan) is None


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("donor_ligand", ("O", "C"))
@pytest.mark.parametrize("substituent", (None, "C", "F"))
def test_graph_family_roundtrips_without_changing_parent_chemistry(donor_ligand, substituent):
    source = _graph(donor_ligand, substituent)
    _assert_roundtrip(source)


def _assert_roundtrip(source):
    names = set()
    for order in _orders(source):
        graph = Chem.RenumberAtoms(source, order)
        mol, plan = _plan(graph)
        assert charged_donor_citation_pi_dead_end(mol, plan) is None
        assert len(plan.ast.component_occurrences) == 4
        assert not plan.derivative_state.hydro_operations
        assert not plan.derivative_state.unsaturation_operations
        named = name_mol(graph, include_trace=True, verify_self=True)
        assert named.error is None
        assert named.parent_nomenclature == "systematic_fusion"
        assert not named.self_audit.coverage.unnamed_atoms
        check = verify_with_opsin(named.name, Chem.MolToSmiles(source), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip
        names.add(named.name)
    assert len(names) == 1


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_substituted_aryl_n_oxide_roundtrip():
    _assert_roundtrip(Chem.MolFromSmiles("COc1ccc(-c2nc3cc4c(cc3n2)[n+]([O-])c2ccccc2n4O)cc1"))
