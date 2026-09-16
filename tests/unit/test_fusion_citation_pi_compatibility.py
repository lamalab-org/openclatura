"""Choose an audited citation whose construction can realize its carbon H."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion import planner
from openclatura.fusion.citation_pi import citation_pi_dead_end
from openclatura.fusion.mancude import indicated_hydrogen_parent_bond_model
from openclatura.fusion.model import FusionConfirmed
from openclatura.graph_io import read_rdkit_mol
from openclatura.opsin_verify import verify_with_opsin

SMILES = "CC1=CC(Cl)=NC2=Nc3ccccc3CC2=N1"


def _orders(size):
    original = list(range(size))
    yield original
    yield original[::-1]
    for seed in (7, 73, 48799, 20483):
        shuffled = original.copy()
        random.Random(seed).shuffle(shuffled)
        yield shuffled


def _plan(graph):
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = planner.plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    return mol, result.plan


def test_rejected_citation_has_a_proved_uncompletable_first_edge(monkeypatch):
    graph = Chem.MolFromSmiles(SMILES)
    for order in _orders(graph.GetNumAtoms()):
        with monkeypatch.context() as context:
            context.setattr(planner, "citation_pi_dead_end", lambda plan: None)
            mol, plan = _plan(Chem.RenumberAtoms(graph, order))
        edge = citation_pi_dead_end(plan)
        assert edge is not None
        assert mol.get_bond(*edge).order == 1
        assert all(dict(assignment.orders)[edge] == 1 for assignment in plan.bond_model.allowed_kekule_assignments)
        assert plan.audit.confirmed
        assert not plan.derivative_state.hydro_operations
        assert not plan.derivative_state.unsaturation_operations
        assert len(plan.indicated_hydrogens) == 1
        assert citation_pi_dead_end(replace(plan, indicated_hydrogens=())) is None


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize(
    "halogen,chain_length,branched",
    [(9, 0, False), (17, 0, False), (35, 0, False), (53, 0, False), (17, 1, False), (17, 2, False), (17, 2, True)],
)
def test_graph_modified_citations_roundtrip_in_every_atom_order(halogen, chain_length, branched):
    graph = Chem.RWMol(Chem.MolFromSmiles(SMILES))
    chlorine = next(atom for atom in graph.GetAtoms() if atom.GetAtomicNum() == 17)
    chlorine.SetAtomicNum(halogen)
    terminal = next(atom.GetIdx() for atom in graph.GetAtoms() if atom.GetAtomicNum() == 6 and not atom.IsInRing())
    current = terminal
    for _ in range(chain_length):
        added = graph.AddAtom(Chem.Atom(6))
        graph.AddBond(terminal if branched else current, added, Chem.BondType.SINGLE)
        current = added
    Chem.SanitizeMol(graph)
    smiles = Chem.MolToSmiles(graph)
    names = set()
    for order in _orders(graph.GetNumAtoms()):
        permuted = Chem.RenumberAtoms(graph, order)
        _, plan = _plan(permuted)
        assert len(plan.ast.component_occurrences) == 3
        assert citation_pi_dead_end(plan) is None
        assert plan.audit.confirmed
        result = name_mol(permuted, include_trace=True, verify_self=True, token_debug=True)
        assert result.error is None
        assert result.parent_nomenclature == "systematic_fusion"
        assert not result.self_audit.coverage.unnamed_atoms
        assert "11H-" in result.name
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip
        names.add(result.name)
    assert len(names) == 1


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize(
    "smiles", ["CC1=Nc2cc3ccccc3nc2N=C(Cl)C1", "CC1=Nc2cc3c(nc2=NC(Cl)=C1)CC=CC=3", "CC1=Nc2cc3c(nc2=NC(Cl)=C1)C=CCC=3"]
)
def test_valid_first_pi_choice_keeps_preferred_fusion_citation(smiles):
    graph = Chem.MolFromSmiles(smiles)
    names = set()
    for order in _orders(graph.GetNumAtoms()):
        permuted = Chem.RenumberAtoms(graph, order)
        _, plan = _plan(permuted)
        assert len(plan.ast.component_occurrences) == 2
        by_locant = {locant: atom for atom, locant in plan.numbering.string_input_locant_maps()[0].items()}
        first_edge = tuple(sorted((by_locant["1"], by_locant["2"])))
        assert any(
            dict(assignment.orders)[first_edge] == 2 for assignment in plan.bond_model.allowed_kekule_assignments
        )
        assert citation_pi_dead_end(plan) is None
        result = name_mol(permuted, include_trace=True, verify_self=True)
        assert result.error is None
        assert result.parent_nomenclature == "systematic_fusion"
        assert not result.self_audit.coverage.unnamed_atoms
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip
        names.add(result.name)
    assert len(names) == 1


@pytest.mark.parametrize("outside_scope", ["charged", "unpaired_donor"])
def test_charged_and_unpaired_donor_states_are_not_rejected(monkeypatch, outside_scope):
    with monkeypatch.context() as context:
        context.setattr(planner, "citation_pi_dead_end", lambda plan: None)
        _, parent = _plan(Chem.MolFromSmiles(SMILES))
    plan = parent.numbering_variants[0]
    assert citation_pi_dead_end(plan) is not None
    nitrogen = next(atom.id for atom in plan.abstract_parent_graph.atoms if atom.symbol == "N")
    if outside_scope == "charged":
        graph = replace(
            plan.abstract_parent_graph,
            atoms=tuple(
                replace(atom, formal_charge=1) if atom.id == nitrogen else atom
                for atom in plan.abstract_parent_graph.atoms
            ),
        )
        plan = replace(plan, abstract_parent_graph=graph)
    else:
        indicated = {atom for atom, locant in plan.numbering.input_locant_maps[0] if locant in plan.indicated_hydrogens}
        model = indicated_hydrogen_parent_bond_model(plan.abstract_parent_graph, indicated | {nitrogen})
        assert model.maximum_non_cumulative_double_bonds < plan.bond_model.maximum_non_cumulative_double_bonds
        plan = replace(plan, bond_model=model)
    assert citation_pi_dead_end(plan) is None


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_nitrogen_h_tautomer_keeps_the_preferred_two_component_citation():
    smiles = "N1=CC=CN=C2NC=3C=CC=CC3C=C21"
    graph = Chem.MolFromSmiles(smiles)
    for order in _orders(graph.GetNumAtoms()):
        permuted = Chem.RenumberAtoms(graph, order)
        _, plan = _plan(permuted)
        assert len(plan.ast.component_occurrences) == 2
        assert citation_pi_dead_end(plan) is None
        result = name_mol(permuted, include_trace=True, verify_self=True)
        assert result.error is None
        assert not result.self_audit.coverage.unnamed_atoms
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip
