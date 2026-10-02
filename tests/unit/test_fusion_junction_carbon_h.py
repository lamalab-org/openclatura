"""Graph-built witnesses for consumption of movable junction carbon H."""

from dataclasses import replace

import pytest
from test_fusion_audit import _two_fused_rings

from openclatura.fusion import indicated_hydrogen as intrinsic
from openclatura.fusion.component_hydrogen import component_hydrogen_consumption
from openclatura.fusion.numbering import parent_bond_model


def _junction_case(aromatic):
    # Two five-rings expose a changed completed pi budget. A five/six pair
    # supplies the aromatic witness with an explicitly Kekulized six-ring.
    case = _two_fused_rings(5, 6 if aromatic else 5)
    junction = 3
    specs = {}
    for match in case.ast.component_occurrences:
        spec = case.registry[match.spec_key]
        default = next(locant for locant, atom in match.local_to_input_atom if atom == junction)
        owns_h = match.occurrence_id == 0
        template = replace(
            spec.template,
            default_indicated_h=(default,) if owns_h else (),
            mancude_double_bonds=2 if owns_h or not aromatic else 3,
            atoms=tuple(
                replace(atom, saturated=owns_h and atom.locant == default, default_h=owns_h and atom.locant == default)
                for atom in spec.atoms
            ),
            bonds=tuple(replace(bond, bond_class="mancude") for bond in spec.bonds),
        )
        specs[match.occurrence_id] = replace(spec, template=template)
    doubles = {(0, 1), (3, 4), (5, 6), (7, 8)} if aromatic else {(0, 1), (2, 3), (4, 5), (6, 7)}
    for bond in case.mol.bonds.values():
        case.mol.update_bond(bond.idx, order=2 if tuple(sorted((bond.u, bond.v))) in doubles else 1)
    for atom in case.mol.atoms:
        valence = sum(case.mol.get_bond(atom, other).order for other in case.mol.get_neighbors(atom))
        case.mol.update_atom(atom, total_h_count=4 - valence, is_aromatic=aromatic and atom >= 3)
    return case, specs, junction


def _project(case, specs):
    candidates = intrinsic.intrinsic_carbon_candidate_atoms(case.ast, specs, case.mol)
    graph = intrinsic.component_parent_graph(case.ast, specs, relocate_carbon_h=False)
    sites = intrinsic.pi_bearing_fusion_carbon_sites(case.mol, graph, candidates)
    return graph, candidates, sites


@pytest.mark.parametrize("aromatic", [False, True], ids=["kekule", "aromatic"])
def test_declared_movable_pi_junction_releases_only_its_local_h_constraint(aromatic):
    case, specs, junction = _junction_case(aromatic)
    original_template = specs[0].template
    relaxed = intrinsic.component_parent_atoms(specs[0])
    assert relaxed != specs[0].atoms
    graph, candidates, sites = _project(case, specs)
    assert junction in candidates
    assert sites == frozenset({junction})
    assert not intrinsic.pi_bearing_fusion_carbon_sites(case.mol, graph, frozenset())
    original = parent_bond_model(graph)
    model, carbon_h = intrinsic.intrinsic_carbon_parent_model(
        case.mol, graph, original, dict(case.numbering.input_locant_maps[0]), candidates
    )
    incident = frozenset(edge for edge in original.required_single_bonds if junction in edge)
    assert len(incident) == 3
    assert model.required_single_bonds == (frozenset({(1, 2), (2, 3)}) if aromatic else frozenset())
    assert carbon_h == (frozenset({2}) if aromatic else frozenset())
    intrinsic_single = frozenset(bond.atoms for bond in graph.bonds if carbon_h.intersection(bond.atoms))
    assert model.required_single_bonds == (original.required_single_bonds - incident) | intrinsic_single
    assert model.allowed_kekule_assignments
    assert all(
        sum(order == 2 for edge, order in assignment.orders if junction in edge) == 1
        for assignment in model.allowed_kekule_assignments
    )
    assert model.maximum_non_cumulative_double_bonds == 4
    if not aromatic:
        assert original.maximum_non_cumulative_double_bonds == 3
        # Relaxing the junction changes the completed pi budget (3 -> 4). That
        # is the point rather than an objection: P-25.7.1.3 assigns indicated
        # hydrogen to the completed system, so the fused parent's budget is
        # meant to be recomputed once the component's own tautomer is released.
        # Here fusion is additionally *proved* to consume the released H at the
        # shared junction.
        budget = {
            relaxed: parent_bond_model(
                intrinsic.component_parent_graph(case.ast, specs, relocate_carbon_h=relaxed)
            ).maximum_non_cumulative_double_bonds
            for relaxed in (False, True)
        }
        assert budget[False] != budget[True]
        assert component_hydrogen_consumption(case.ast, specs) is not None
        assert intrinsic.component_carbon_h_relocation_scope(case.ast, specs)
    assert specs[0].template is original_template


@pytest.mark.parametrize("aromatic", [False, True], ids=["kekule", "aromatic"])
@pytest.mark.parametrize(
    "restriction", ["undeclared_h", "shared_saturation", "shared_zero_capacity", "shared_forced_single"]
)
def test_pi_junction_cannot_override_undeclared_or_inherited_roles(aromatic, restriction):
    case, specs, junction = _junction_case(aromatic)
    occurrence = 0 if restriction == "undeclared_h" else 1
    spec = specs[occurrence]
    match = next(match for match in case.ast.component_occurrences if match.occurrence_id == occurrence)
    locant = next(locant for locant, atom in match.local_to_input_atom if atom == junction)
    if restriction == "undeclared_h":
        template = replace(spec.template, default_indicated_h=())
    else:
        changes = {
            "shared_saturation": {"saturated": True},
            "shared_zero_capacity": {"pi_capacity": 0},
            "shared_forced_single": {"forced_single": True},
        }[restriction]
        template = replace(
            spec.template,
            atoms=tuple(replace(atom, **changes) if atom.locant == locant else atom for atom in spec.atoms),
        )
    specs[occurrence] = replace(spec, template=template)
    graph, candidates, sites = _project(case, specs)
    assert junction not in candidates
    assert not sites
    original = parent_bond_model(graph)
    model, _ = intrinsic.intrinsic_carbon_parent_model(
        case.mol, graph, original, dict(case.numbering.input_locant_maps[0]), candidates
    )
    incident = frozenset(edge for edge in original.required_single_bonds if junction in edge)
    assert len(incident) == 3
    assert incident <= model.required_single_bonds


@pytest.mark.parametrize("external_order", [None, 2, 3], ids=["saturated", "external_double", "external_triple"])
def test_saturated_junction_and_external_pi_are_not_internal_pi_witnesses(external_order):
    case, specs, junction = _junction_case(False)
    for neighbor in case.mol.get_neighbors(junction):
        case.mol.update_bond(case.mol.get_bond(junction, neighbor).idx, order=1)
    case.mol.update_atom(junction, total_h_count=1)
    if external_order is not None:
        # Adversarial external pi must not qualify, even with a supplied
        # movable candidate. Normal graph-valence auditing remains separate.
        case.mol.add_atom("C", idx=99)
        case.mol.add_bond(junction, 99, order=external_order, idx=999)
    graph, _candidates, sites = _project(case, specs)
    assert not sites
    assert not intrinsic.pi_bearing_fusion_carbon_sites(case.mol, graph, frozenset({junction}))


def test_internal_triple_is_not_a_movable_junction_double_bond():
    case, specs, junction = _junction_case(False)
    case.mol.update_bond(case.mol.get_bond(2, junction).idx, order=3)
    graph = intrinsic.component_parent_graph(case.ast, specs, relocate_carbon_h=False)
    assert not intrinsic.pi_bearing_fusion_carbon_sites(case.mol, graph, frozenset({junction}))


@pytest.mark.parametrize(
    "change", [{"symbol": "N"}, {"formal_charge": 1}, {"forced_single": True}, {"saturated": False}]
)
def test_supplied_candidate_does_not_bypass_component_atom_domain(change):
    case, specs, junction = _junction_case(False)
    graph = intrinsic.component_parent_graph(case.ast, specs, relocate_carbon_h=False)
    graph = replace(
        graph, atoms=tuple(replace(atom, **change) if atom.id == junction else atom for atom in graph.atoms)
    )
    assert not intrinsic.pi_bearing_fusion_carbon_sites(case.mol, graph, frozenset({junction}))


@pytest.mark.parametrize("aromatic", [False, True], ids=["kekule", "aromatic"])
@pytest.mark.parametrize("restriction", ["charge", "heteroatom", "degree_two", "degree_four"])
def test_observed_junction_must_be_neutral_carbon_with_three_parent_neighbors(aromatic, restriction):
    case, specs, junction = _junction_case(aromatic)
    graph = intrinsic.component_parent_graph(case.ast, specs, relocate_carbon_h=False)
    mol = case.mol
    if restriction == "charge":
        mol.update_atom(junction, charge=1)
    elif restriction == "heteroatom":
        mol.update_atom(junction, symbol="N")
    elif restriction == "degree_two":
        mol = mol.subgraph(set(mol.atoms) - {2})
        assert len(set(mol.atoms).intersection(mol.get_neighbors(junction))) == 2
    else:
        mol.add_bond(junction, 0, idx=999)
        assert len(set(mol.atoms).intersection(mol.get_neighbors(junction))) == 4
    assert not intrinsic.pi_bearing_fusion_carbon_sites(mol, graph, frozenset({junction}))
