"""Fixed sigma valence is not an unpaired carbon-H obligation."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.indicated_hydrogen import intrinsic_carbon_candidate_atoms, intrinsic_carbon_parent_model
from openclatura.fusion.model import FusionConfirmed, FusionGraph, FusionGraphAtom, FusionGraphBond, FusionMode
from openclatura.fusion.numbering import parent_bond_model, parent_pi_capable_atom_ids
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol


def _graph(doubles=((1, 2), (3, 4))):
    graph = Chem.RWMol()
    for atom in range(9):
        graph.AddAtom(Chem.Atom("O" if atom == 7 else "C"))
    edges = [(atom, (atom + 1) % 9) for atom in range(9)] + [(2, 6), (6, 8)]
    multiple = {frozenset(edge) for edge in doubles}
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if frozenset(edge) in multiple else Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


def test_four_sigma_carbon_does_not_hide_a_separate_ch2_site():
    mol = read_rdkit_mol(_graph())
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    graph = plan.abstract_parent_graph
    capable = parent_pi_capable_atom_ids(graph)
    assert 6 not in capable
    assert 5 in capable
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    candidates = intrinsic_carbon_candidate_atoms(plan.ast, specs, mol)
    model, sites = intrinsic_carbon_parent_model(
        mol, graph, parent_bond_model(graph), dict(plan.numbering.input_locant_maps[0]), candidates
    )
    assert sites == frozenset({5})
    assert model.maximum_non_cumulative_double_bonds == 3
    assert plan.bond_model == model
    assert tuple(map(str, plan.indicated_hydrogens)) == ("6",)
    assert all(
        order == 1 for assignment in model.allowed_kekule_assignments for edge, order in assignment.orders if 6 in edge
    )


@pytest.mark.parametrize("fixed_double", [False, True])
def test_pi_capacity_respects_mandatory_load_without_losing_available_carbon_valence(fixed_double):
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C" if atom == 0 else "O") for atom in range(4)),
        bonds=tuple(
            FusionGraphBond((0, atom), "double" if fixed_double and atom == 1 else "single") for atom in range(1, 4)
        ),
    )
    assert (0 in parent_pi_capable_atom_ids(graph)) is not fixed_double
    # No eligible edge does not imply a carbon has exhausted its sigma valence.
    assert not parent_bond_model(graph).pi_eligible_edges


def test_fixed_valence_constraint_cannot_be_overridden_by_template_pi_flag():
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in range(5)),
        bonds=tuple(FusionGraphBond((0, atom)) for atom in range(1, 5)),
    )
    assert 0 not in parent_pi_capable_atom_ids(graph)
    assert not parent_bond_model(graph).pi_eligible_edges
    invalid = replace(
        graph, atoms=graph.atoms + (FusionGraphAtom(5, "C"),), bonds=graph.bonds + (FusionGraphBond((0, 5)),)
    )
    with pytest.raises(ValueError, match="fixed-valence limit"):
        parent_pi_capable_atom_ids(invalid)


@pytest.mark.opsin
@pytest.mark.parametrize(
    "doubles",
    [((1, 2), (3, 4)), ((1, 2), (4, 5)), ((1, 2),), ()],
)
def test_graph_built_fixed_junction_family_is_public_fusion_exact(doubles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _graph(doubles)
    original = Chem.MolToSmiles(graph)
    names = []
    for order in (list(range(9)), list(reversed(range(9)))):
        reordered = Chem.RenumberAtoms(graph, order)
        mol = read_rdkit_mol(reordered)
        planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
        assert isinstance(planned, FusionConfirmed), planned
        result = name_mol(reordered)
        assert result, result
        assert "pentaleno[1,6a-b]oxirene" in result.name
        check = verify_with_opsin(result.name, original, standardize_smiles=False)
        assert check.ok, check.to_dict()
        assert check.canonical_roundtrip == original, check.to_dict()
        names.append(result.name)
    assert names[0] == names[1]
