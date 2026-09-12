"""Graph-only regression coverage for intrinsic sites and derivative replay."""

from dataclasses import replace

import pytest
from test_fusion_audit import _two_fused_rings

from openclatura.chains import find_ring_systems
from openclatura.fusion import indicated_hydrogen as intrinsic
from openclatura.fusion.audit import _audit_derivative_state, _has_consistent_derivative_operations
from openclatura.fusion.mancude import parent_derivative_state
from openclatura.fusion.model import FusionConfirmed, FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import MancudeSearchBudgetExceeded, parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_smiles
from openclatura.locants import SystemLocant
from openclatura.molecule import Molecule


def _two_carbon_h_domains():
    # Two odd pi domains joined by a fixed single edge. Each maximum matching
    # leaves one CH2; neither site may be obtained by deleting a parent double.
    edges = tuple(tuple(sorted((offset + atom, offset + (atom + 1) % 5))) for offset in (0, 5) for atom in range(5)) + (
        (2, 7),
    )
    doubles = {(1, 2), (3, 4), (6, 7), (8, 9)}
    mol = Molecule()
    for atom in range(10):
        mol.add_atom("C", idx=atom, total_h_count=2 if atom in {0, 5} else 0 if atom in {2, 7} else 1)
    for bond_id, edge in enumerate(edges):
        mol.add_bond(*edge, idx=bond_id, order=2 if edge in doubles else 1)
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in range(10)),
        bonds=tuple(FusionGraphBond(edge, "single" if edge == (2, 7) else "mancude") for edge in edges),
    )
    locants = {atom: SystemLocant(atom + 1) for atom in range(10)}
    return mol, graph, locants


def test_multiple_intrinsic_carbon_sites_preserve_joint_maximum():
    mol, graph, locants = _two_carbon_h_domains()
    original = parent_bond_model(graph)
    model, sites = intrinsic.intrinsic_carbon_parent_model(mol, graph, original, locants, frozenset({0, 5}))
    assert sites == frozenset({0, 5})
    assert model.maximum_non_cumulative_double_bonds == original.maximum_non_cumulative_double_bonds == 4
    assert all(
        order == 1
        for assignment in model.allowed_kekule_assignments
        for edge, order in assignment.orders
        if sites.intersection(edge)
    )
    state = parent_derivative_state(mol, frozenset(locants), model, locants)
    assert state is not None
    assert not state.hydro_operations
    assert not state.unsaturation_operations


def test_unproved_unpaired_carbon_cannot_be_silently_ignored():
    mol, graph, locants = _two_carbon_h_domains()
    original = parent_bond_model(graph)
    model, sites = intrinsic.intrinsic_carbon_parent_model(mol, graph, original, locants, frozenset({0}))
    assert not sites
    assert model == original


def test_oxo_converts_joint_carbon_h_sites_without_deleting_parent_pi_bonds():
    mol, graph, locants = _two_carbon_h_domains()
    mol.update_atom(0, total_h_count=0)
    mol.add_atom("O", idx=10)
    mol.add_bond(0, 10, idx=11, order=2)
    original = parent_bond_model(graph)
    model, sites = intrinsic.intrinsic_carbon_parent_model(mol, graph, original, locants, frozenset({0, 5}))
    assert not sites
    assert model.maximum_non_cumulative_double_bonds == original.maximum_non_cumulative_double_bonds == 4
    state = parent_derivative_state(mol, frozenset(locants), model, locants)
    assert state is not None
    assert not state.hydro_operations
    assert not state.added_hydrogen_operations
    assert not state.unsaturation_operations
    (operation,) = state.intrinsic_hydro_operations
    assert operation.locants == ("1", "6")
    assert operation.atom_ids == (0, 5)
    assert operation.bond_ids == ()
    assert state.oxo_operations[0].parent_atom_id == 0


def test_carbon_site_constraint_propagates_matching_budget(monkeypatch):
    mol, graph, locants = _two_carbon_h_domains()
    original = parent_bond_model(graph)

    def exhausted(*args, **kwargs):
        raise MancudeSearchBudgetExceeded(1)

    monkeypatch.setattr(intrinsic, "_single_site_parent_model", exhausted)
    with pytest.raises(MancudeSearchBudgetExceeded):
        intrinsic.intrinsic_carbon_parent_model(mol, graph, original, locants, frozenset({0, 5}))


@pytest.mark.parametrize("substituted", (False, True))
def test_shared_lone_pair_helper_preserves_charged_oxo_derivative_donor(substituted):
    mol = Molecule()
    mol.add_atom("N", idx=0, is_aromatic=True, total_h_count=0 if substituted else 1)
    mol.add_atom("C", idx=1, charge=-1, is_aromatic=True)
    mol.add_atom("C", idx=2, is_aromatic=True)
    mol.add_atom("O", idx=3)
    mol.add_bond(0, 1, idx=0)
    mol.add_bond(0, 2, idx=1)
    mol.add_bond(1, 2, idx=2)
    mol.add_bond(2, 3, idx=3, order=2)
    if substituted:
        mol.add_atom("C", idx=4, total_h_count=3)
        mol.add_bond(0, 4, idx=4)
    graph = FusionGraph(
        atoms=(FusionGraphAtom(0, "N"), FusionGraphAtom(1, "C"), FusionGraphAtom(2, "C")),
        bonds=tuple(FusionGraphBond(edge) for edge in ((0, 1), (0, 2), (1, 2))),
    )
    assert not intrinsic.aromatic_nitrogen_hydrogen_atoms(mol, graph)
    assert intrinsic.intrinsic_parent_lone_pair_sites(mol, graph) == frozenset({0})
    mol.update_atom(0, charge=1)
    assert not intrinsic.intrinsic_parent_lone_pair_sites(mol, graph)


def test_operation_consistency_does_not_require_component_owned_hydro():
    case = _two_fused_rings()
    locants = dict(case.numbering.input_locant_maps[0])
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in sorted(case.parent_atoms)),
        bonds=tuple(FusionGraphBond(bond.atoms) for bond in case.graph.bonds),
    )
    state = parent_derivative_state(case.mol, case.parent_atoms, parent_bond_model(graph), locants)
    assert state is not None and state.hydro_operations
    assert _has_consistent_derivative_operations(case.mol, case.ast, {}, case.numbering, (), state)
    operation = state.hydro_operations[0]
    duplicate_owner = replace(state, bond_delta=replace(state.bond_delta, added_hydrogen_operations=(operation,)))
    assert not _has_consistent_derivative_operations(case.mol, case.ast, {}, case.numbering, (), duplicate_owner)


def test_duplicate_oxo_is_rejected_by_independent_replay():
    case = _two_fused_rings()
    mol = case.mol
    parent_atom = 0
    oxygen = max(mol.atoms) + 1
    mol.add_atom("O", idx=oxygen)
    mol.add_bond(parent_atom, oxygen, idx=max(mol.bonds) + 1, order=2)
    state = parent_derivative_state(mol, case.parent_atoms, case.bond_model, dict(case.numbering.input_locant_maps[0]))
    assert state is not None and len(state.oxo_operations) == 1
    errors = []
    _audit_derivative_state(
        mol,
        case.parent_atoms,
        case.numbering,
        case.bond_model,
        (),
        replace(state, oxo_operations=state.oxo_operations * 2),
        errors,
    )
    assert "typed oxo operations duplicate an exocyclic parent oxo group" in errors


@pytest.mark.parametrize(
    "smiles",
    (
        "C=C1C(=O)O[C@H]2[C@H]1CCC(C)=C1CCC(=O)O[C@]12C",
        "O=S1(=O)CCc2c1scc/c2=N\\NC",
    ),
)
@pytest.mark.parametrize("corruption", ("empty_hydro", "overlapping_added_h", "outside_parent"))
def test_external_derivative_composition_requires_nonempty_independent_hydro(smiles, corruption):
    mol = read_smiles(smiles)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    state = plan.derivative_state
    assert _has_consistent_derivative_operations(mol, plan.ast, specs, plan.numbering, (), state)
    if corruption == "empty_hydro":
        state = replace(state, hydro_operations=())
    elif corruption == "overlapping_added_h":
        state = replace(state, bond_delta=replace(state.bond_delta, added_hydrogen_operations=state.hydro_operations))
    else:
        state = replace(state, hydro_operations=(replace(state.hydro_operations[0], atom_ids=(-1, -2)),))
    assert not _has_consistent_derivative_operations(mol, plan.ast, specs, plan.numbering, (), state)


def test_higher_order_join_does_not_admit_an_unproved_external_multiple_bond():
    mol = read_smiles("C=C1C(=O)O[C@H]2[C@H]1CCC(C)=C1CCC(=O)O[C@]12C")
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    assert _has_consistent_derivative_operations(mol, plan.ast, specs, plan.numbering, (), plan.derivative_state)
    mol.update_atom(0, symbol="S", total_h_count=0)
    assert not _has_consistent_derivative_operations(mol, plan.ast, specs, plan.numbering, (), plan.derivative_state)


@pytest.mark.parametrize("corruption", ("charged", "triple", "valence", "hydro_overlap"))
def test_component_owned_hydro_cannot_bypass_external_bond_checks(corruption):
    mol = read_smiles("O=S1(=O)CCc2c1scc/c2=N\\NC")
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    occurrences = [atom for match in plan.ast.component_occurrences for atom in match.input_atom_by_locant.values()]
    scopes = [
        {
            match.input_atom_by_locant[atom.locant]
            for atom in specs[match.occurrence_id].atoms
            if atom.symbol == "C" and occurrences.count(match.input_atom_by_locant[atom.locant]) == 1
        }
        for match in plan.ast.component_occurrences
    ]
    state = plan.derivative_state
    assert state.hydro_operations
    assert all(set(operation.atom_ids) in scopes for operation in state.hydro_operations)
    assert _has_consistent_derivative_operations(mol, plan.ast, specs, plan.numbering, (), state)
    ((parent, external),) = [
        (atom, other)
        for atom in atoms
        for other in mol.get_neighbors(atom)
        if other not in atoms and mol.atoms[other].symbol == "N" and mol.get_bond(atom, other).order == 2
    ]
    if corruption == "charged":
        mol.update_atom(external, charge=1)
    elif corruption == "triple":
        mol.update_bond(mol.get_bond(parent, external).idx, order=3)
    elif corruption == "valence":
        mol.update_atom(external, total_h_count=1)
    else:
        carbon = max(mol.atoms) + 1
        mol.add_atom("C", idx=carbon, total_h_count=2)
        atom = state.hydro_operations[0].atom_ids[0]
        mol.update_atom(atom, total_h_count=0)
        mol.add_bond(atom, carbon, order=2, idx=max(mol.bonds) + 1)
    assert not _has_consistent_derivative_operations(mol, plan.ast, specs, plan.numbering, (), state)


def test_aromatic_polycomponent_parent_exposes_no_pi_bearing_junction():
    """This benzopyranopyrrole plans as a three-component tree with no movable junction.

    It used to be the witness for the "a junction may not release an unproved
    component role" guard, but the parent is now composed from pyrrole, pyran and
    benzene, and none of its carbon candidates carries a pi-bearing junction
    role -- a scan of the 5000-molecule OPSIN-verified corpus finds zero
    confirmed fusion plans with such a site. That guard is exercised on the
    synthetic two-ring fixture instead, by
    ``test_pi_junction_cannot_override_undeclared_or_inherited_roles`` in
    ``test_fusion_junction_carbon_h.py``. What this molecule still pins is the
    composition itself and the name it produces.
    """

    mol = read_smiles("CSc1ccc(C2c3c(oc4ccccc4c3=O)C(=O)N2c2ncccn2)cc1")
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    assert not plan.derivative_state.unsaturation_operations
    assert plan.ast.plan_kind == "polycomponent_tree"
    assert [match.spec_key for match in plan.ast.component_occurrences] == ["pyrrole", "pyran", "benzene"]

    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    candidates = intrinsic.intrinsic_carbon_candidate_atoms(plan.ast, specs, mol)
    assert candidates
    assert not intrinsic.pi_bearing_fusion_carbon_sites(mol, plan.abstract_parent_graph, candidates)
    assert not intrinsic.pi_bearing_fusion_carbon_sites(mol, plan.abstract_parent_graph, frozenset())


@pytest.mark.opsin
def test_aromatic_polycomponent_parent_name_roundtrips():
    from rdkit import Chem

    from openclatura import name_mol, opsin_available, verify_with_opsin

    if not opsin_available():
        pytest.skip("py2opsin/Java is unavailable")
    smiles = "CSc1ccc(C2c3c(oc4ccccc4c3=O)C(=O)N2c2ncccn2)cc1"
    graph = Chem.MolFromSmiles(smiles)
    result = name_mol(graph, include_trace=True)
    assert result.error is None
    assert result.name == (
        "1-(4-(methylsulfanyl)phenyl)-2-(pyrimidin-2-yl)-1H-benzo[1',2':2,3]pyrano[5,6-c]pyrrole-3,9-dione"
    )
    check = verify_with_opsin(result.name, Chem.MolToSmiles(graph), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()


def test_intrinsic_carbon_scope_accepts_ortho_peri_but_requires_pi_budgets():
    case = _two_fused_rings(interface_atom_count=3)
    specs = {
        match.occurrence_id: replace(
            case.registry[match.spec_key],
            template=replace(case.registry[match.spec_key].template, mancude_double_bonds=0),
        )
        for match in case.ast.component_occurrences
    }
    assert intrinsic.intrinsic_carbon_fusion_scope(case.ast, specs)
    missing_budget = {**specs, 0: replace(specs[0], template=replace(specs[0].template, mancude_double_bonds=None))}
    assert not intrinsic.intrinsic_carbon_fusion_scope(case.ast, missing_budget)
    assert not intrinsic.intrinsic_carbon_fusion_scope(case.ast, {0: specs[0]})
