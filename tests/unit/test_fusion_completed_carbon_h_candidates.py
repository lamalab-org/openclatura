"""Completed matching can require carbon H absent from isolated components."""

from dataclasses import replace

import pytest
from rdkit import Chem
from test_fusion_audit import _two_fused_rings

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion import indicated_hydrogen as intrinsic
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import AuditStatus, FusionConfirmed
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.molecule import Molecule


def _fused_furans():
    case = _two_fused_rings(5, 5)
    oxygens = {1, 6}
    for atom in case.mol.atoms:
        case.mol.update_atom(
            atom,
            symbol="O" if atom in oxygens else "C",
            total_h_count=0 if atom in oxygens else 4 - len(case.mol.get_neighbors(atom)),
        )
    specs = {}
    for match in case.ast.component_occurrences:
        spec = case.registry[match.spec_key]
        template = replace(
            spec.template,
            mancude_double_bonds=2,
            atoms=tuple(
                replace(
                    atom, symbol="O" if match.input_atom_by_locant[atom.locant] in oxygens else "C", saturated=False
                )
                for atom in spec.atoms
            ),
            bonds=tuple(replace(bond, bond_class="mancude") for bond in spec.bonds),
        )
        specs[match.occurrence_id] = replace(spec, template=template)
    return case, specs


def test_completed_matching_finds_unpaired_carbons_with_no_local_h_candidates():
    case, specs = _fused_furans()
    assert all(not intrinsic._component_carbon_h_locants(spec) for spec in specs.values())
    assert intrinsic.intrinsic_carbon_fusion_scope(case.ast, specs)
    assert intrinsic.component_carbon_h_relocation_scope(case.ast, specs)
    candidates = intrinsic.intrinsic_carbon_candidate_atoms(case.ast, specs, case.mol)
    # Candidates are component pi-capable roles, not the H assignment.
    assert candidates == {0, 2, 3, 4, 5, 7}
    eligible_h = {
        atom for atom in candidates if intrinsic.is_intrinsic_carbon_h_site(case.mol, atom, case.parent_atoms)
    }
    assert eligible_h == {0, 2, 3, 4, 5, 7}
    graph = intrinsic.component_parent_graph(case.ast, specs, relocate_carbon_h=True)
    original = parent_bond_model(graph)
    model, sites = intrinsic.intrinsic_carbon_parent_model(
        case.mol, graph, original, dict(case.numbering.input_locant_maps[0]), candidates
    )
    assert model.maximum_non_cumulative_double_bonds == original.maximum_non_cumulative_double_bonds == 2
    assert len(sites) == 2
    # Saturated junctions are candidates too, but this matching still proves
    # only the original nonjunction H sites.
    assert sites <= {0, 2, 5, 7}
    assert all(intrinsic.is_intrinsic_carbon_h_site(case.mol, atom, case.parent_atoms) for atom in sites)
    assert all(
        frozenset(atom for edge, order in assignment.orders if order == 2 for atom in edge)
        == {0, 2, 3, 4, 5, 7} - sites
        for assignment in model.allowed_kekule_assignments
    )


def test_completed_fully_paired_carbon_roles_do_not_assign_carbon_h():
    case = _two_fused_rings(6, 6)
    for atom in case.mol.atoms:
        case.mol.update_atom(atom, total_h_count=4 - len(case.mol.get_neighbors(atom)))
    specs = {match.occurrence_id: case.registry[match.spec_key] for match in case.ast.component_occurrences}
    specs = {
        occurrence: replace(
            spec,
            template=replace(
                spec.template,
                mancude_double_bonds=3,
                atoms=tuple(replace(atom, saturated=False) for atom in spec.atoms),
                bonds=tuple(replace(bond, bond_class="mancude") for bond in spec.bonds),
            ),
        )
        for occurrence, spec in specs.items()
    }
    assert any(intrinsic.is_intrinsic_carbon_h_site(case.mol, atom, case.parent_atoms) for atom in case.parent_atoms)
    graph = intrinsic.component_parent_graph(case.ast, specs, relocate_carbon_h=True)
    model = parent_bond_model(graph)
    assert model.maximum_non_cumulative_double_bonds == 5
    assert all(
        {atom for edge, order in assignment.orders if order == 2 for atom in edge} == case.parent_atoms
        for assignment in model.allowed_kekule_assignments
    )
    candidates = intrinsic.intrinsic_carbon_candidate_atoms(case.ast, specs, case.mol)
    assert candidates == case.parent_atoms
    selected_model, sites = intrinsic.intrinsic_carbon_parent_model(
        case.mol, graph, model, dict(case.numbering.input_locant_maps[0]), candidates
    )
    assert not sites
    assert selected_model == model


def test_local_candidates_survive_completed_perfect_matching():
    case, specs = _fused_furans()
    for atom in (1, 6):
        case.mol.update_atom(atom, symbol="C", total_h_count=2)
    specs = {
        occurrence: replace(
            spec,
            template=replace(spec.template, atoms=tuple(replace(atom, symbol="C") for atom in spec.atoms)),
        )
        for occurrence, spec in specs.items()
    }
    graph = intrinsic.component_parent_graph(case.ast, specs, relocate_carbon_h=True)
    assert parent_bond_model(graph).maximum_non_cumulative_double_bonds == 4
    local = {
        match.input_atom_by_locant[locant]
        for match in case.ast.component_occurrences
        for locant in intrinsic._component_carbon_h_locants(specs[match.occurrence_id])
    }
    assert local == case.parent_atoms
    assert intrinsic.intrinsic_carbon_candidate_atoms(case.ast, specs, case.mol) == local


@pytest.mark.parametrize("restriction", ["saturated", "forced_single", "pi_capacity"])
def test_shared_component_restriction_blocks_completed_candidate(restriction):
    case, specs = _fused_furans()
    blocked_atom = 3
    shared = [match for match in case.ast.component_occurrences if blocked_atom in match.input_atom_by_locant.values()]
    assert len(shared) == 2
    match = shared[0]
    spec = specs[match.occurrence_id]
    change = {restriction: 0 if restriction == "pi_capacity" else True}
    specs[match.occurrence_id] = replace(
        spec,
        template=replace(
            spec.template,
            atoms=tuple(
                replace(atom, **change) if match.input_atom_by_locant[atom.locant] == blocked_atom else atom
                for atom in spec.atoms
            ),
        ),
    )
    candidates = intrinsic.intrinsic_carbon_candidate_atoms(case.ast, specs, case.mol)
    assert blocked_atom not in candidates
    assert not candidates.intersection({1, 6})


def test_completed_candidates_still_require_all_component_pi_budgets():
    case, specs = _fused_furans()
    specs[0] = replace(specs[0], template=replace(specs[0].template, mancude_double_bonds=None))
    assert not intrinsic.intrinsic_carbon_fusion_scope(case.ast, specs)
    assert not intrinsic.intrinsic_carbon_candidate_atoms(case.ast, specs, case.mol)


def test_inconsistent_declared_component_budget_cannot_supply_pi_candidates():
    _, specs = _fused_furans()
    spec = replace(specs[0], template=replace(specs[0].template, mancude_double_bonds=3))
    assert not intrinsic._component_carbon_pi_locants(spec)


@pytest.mark.parametrize("mode", ["general", "audited_pin"])
@pytest.mark.parametrize("reverse", [False, True])
def test_completed_furan_carbon_h_is_named_and_audited_atom_order_invariant(mode, reverse):
    smiles = "C1OCC2COCC12"
    graph = Chem.MolFromSmiles(smiles)
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    mol = read_rdkit_mol(graph)
    result = plan_fusion_parent(mol, mol.atoms, mode=mode)
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert tuple(map(str, plan.indicated_hydrogens)) == ("1", "3")
    state = plan.derivative_state
    assert len(state.hydro_operations) == 1
    assert state.hydro_operations[0].locants == ("3a", "4", "6", "6a")
    assert not state.unsaturation_operations
    assert not state.intrinsic_hydro_operations
    assert not state.added_hydrogen_operations
    assert plan.bond_model.maximum_non_cumulative_double_bonds == 2
    generated = name_mol(graph, fusion_mode=mode).name
    assert generated == "tetrahydro-1H,3H-furo[3,4-c]furan"
    if opsin_available():
        assert verify_with_opsin(generated, smiles, standardize_smiles=False).status == "matched"
    corrupted = audit_fusion_plan(
        mol,
        frozenset(mol.atoms),
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=(),
        derivative_state=state,
        mode=mode,
    )
    assert corrupted.status is AuditStatus.MISMATCH


@pytest.mark.parametrize("mode", ["general", "audited_pin"])
@pytest.mark.parametrize("reverse", [False, True])
def test_furan_pyrazole_cites_every_intrinsic_carbon_h(mode, reverse):
    smiles = "C1OCC2=NNC=C12"
    graph = Chem.MolFromSmiles(smiles)
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    mol = read_rdkit_mol(graph)
    result = plan_fusion_parent(mol, mol.atoms, mode=mode)
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert tuple(map(str, plan.indicated_hydrogens)) == ("2", "4", "6")
    assert not plan.derivative_state.hydro_operations
    assert not plan.derivative_state.unsaturation_operations
    generated = name_mol(graph, fusion_mode=mode).name
    assert generated == "2H,4H,6H-furo[3,4-c]pyrazole"
    if opsin_available():
        assert verify_with_opsin(generated, smiles, standardize_smiles=False).status == "matched"


def test_graph_imine_pi_cannot_relocate_uncited_carbon_deficit():
    mol = Molecule()
    for atom, (symbol, hydrogens) in enumerate(zip("COCCNNCC", (2, 0, 2, 0, 0, 1, 1, 0))):
        mol.add_atom(symbol, idx=atom, total_h_count=hydrogens, is_aromatic=atom >= 3)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (3, 7)):
        mol.add_bond(left, right, order=2 if (left, right) in {(3, 4), (6, 7)} else 1)
    result = plan_fusion_parent(mol, mol.atoms, mode="general")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    locants = dict(plan.numbering.input_locant_maps[0])
    cited_carbons = {
        atom for atom, locant in locants.items() if locant in plan.indicated_hydrogens and mol.atoms[atom].symbol == "C"
    }
    assert cited_carbons == {0, 2}
    assert not plan.derivative_state.hydro_operations
    assignment = plan.derivative_state.bond_delta.assignment
    paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
    carbons = {atom for atom in mol.atoms if mol.atoms[atom].symbol == "C"}
    assert carbons - paired == cited_carbons
    assert {3, 4, 6, 7} <= paired
