"""Inherited parent eligibility must not force a different fusion descriptor."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.fusion.rules import component_parent_eligible, component_spec_seniority_key
from openclatura.graph_io import read_rdkit_mol

REPORTED_SMILES = "C=CC(O)N1CCN(c2nc(=O)n3c4c(c(-c5ccc(F)cc5)c(Cl)cc24)SC[C@@H](OC)C3)CC1"
CORES = (
    "O=c1ncc2cccc3c2n1CCCS3",
    "O=c1ncc2cccc3c2n1CCCO3",
    "c1cc2cccc3c2n1CCCS3",
)


def _reordered_reported_graph(offset):
    graph = Chem.MolFromSmiles(REPORTED_SMILES)
    order = list(reversed(range(graph.GetNumAtoms())))
    return Chem.RenumberAtoms(graph, order[offset:] + order[:offset])


@pytest.mark.parametrize("mode", (FusionMode.GENERAL, FusionMode.AUDITED_PIN))
def test_peri_indole_selects_the_larger_nitrogen_containing_parent(mode):
    mol = read_rdkit_mol(Chem.MolFromSmiles("N1CC2=CC=CC3=C2C1=CC=C3"))
    planned = plan_fusion_parent(mol, set(mol.atoms), mode=mode)
    assert isinstance(planned, FusionConfirmed), planned
    assert planned.plan.audit.confirmed
    (parent,) = [
        match
        for match in planned.plan.ast.component_occurrences
        if match.occurrence_id in planned.plan.ast.parent_occurrences
    ]
    spec = fusion_component_registry().spec_for_match(parent)
    # P-25.3.2.4: among nitrogen-containing components, more rings wins.
    assert spec.parent_name == "indole"
    assert len(spec.rings) == 2


@pytest.mark.opsin
def test_peri_indole_citation_scopes_hydrogen_to_completed_parent():
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    smiles = "N1CC2=CC=CC3=C2C1=CC=C3"
    result = name_mol(Chem.MolFromSmiles(smiles), include_trace=True)
    assert result.ok, result.error
    assert result.name == "1,2-dihydrobenzo[1,2,3-cd]indole"
    mol = read_rdkit_mol(Chem.MolFromSmiles(smiles))
    plan = plan_fusion_parent(mol, set(mol.atoms), mode=FusionMode.AUDITED_PIN).plan
    sites = {
        str(locant): atom for atom, locant in plan.numbering.input_locant_maps[0] if locant in plan.indicated_hydrogens
    }
    assert {locant: mol.atoms[atom].symbol for locant, atom in sites.items()} == {"1": "N", "2": "C"}
    assert mol.atoms[sites["2"]].total_h_count == 2
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    assert check.ok, check.to_dict()
    assert check.canonical_roundtrip == check.canonical_original


@pytest.mark.parametrize("offset", (0, 7, 19))
@pytest.mark.parametrize("mode", (FusionMode.GENERAL, FusionMode.AUDITED_PIN))
def test_inherited_bicycle_wins_parent_seniority_without_changing_peripheral_numbering(offset, mode):
    graph = _reordered_reported_graph(offset)
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=mode)

    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert plan.audit.confirmed
    assert plan.rendered_base_name == "[1,4]thiazepino[2,3,4-ij]quinazoline"
    registry = fusion_component_registry()
    (parent,) = [
        match for match in plan.ast.component_occurrences if match.occurrence_id in plan.ast.parent_occurrences
    ]
    (attached,) = [
        match for match in plan.ast.component_occurrences if match.occurrence_id not in plan.ast.parent_occurrences
    ]
    parent_spec = registry.spec_for_match(parent)
    attached_spec = registry.spec_for_match(attached)
    assert parent_spec.parent_name == "quinazoline"
    assert not parent_spec.usable_as_parent
    assert parent_spec.usable_as_peri_parent
    assert component_parent_eligible(parent, parent_spec, plan.ast.component_occurrences)
    assert component_parent_eligible(
        parent, replace(parent_spec, rule_reference="independent bibliography"), plan.ast.component_occurrences
    )
    assert not component_parent_eligible(
        parent, replace(parent_spec, usable_as_peri_parent=False), plan.ast.component_occurrences
    )
    assert not component_parent_eligible(parent, parent_spec, (parent,))
    # Even a multi-edge attachment wholly within one constituent ring
    # retains the ordinary attached-only component policy.
    local_atoms = dict(parent.local_to_input_atom)
    same_ring_attachment = replace(
        attached,
        local_to_input_atom=tuple(
            (locant, local_atoms[parent_spec.rings[0][index]] if index < 3 else graph.GetNumAtoms() + index)
            for index, (locant, _) in enumerate(attached.local_to_input_atom)
        ),
    )
    assert not component_parent_eligible(parent, parent_spec, (parent, same_ring_attachment))
    assert component_spec_seniority_key(parent_spec) < component_spec_seniority_key(attached_spec)
    (stereo_atom,) = [atom for atom, _ in Chem.FindMolChiralCenters(graph, includeUnassigned=False)]
    for locant_map in plan.numbering.string_input_locant_maps():
        assert set(locant_map) == atoms
        assert locant_map[stereo_atom] == "3"
        assert {locant_map[atom] for atom in atoms if mol.atoms[atom].symbol == "S"} == {"1"}
        assert {locant_map[atom] for atom in atoms if mol.atoms[atom].symbol == "N"} == {"5", "7"}


@pytest.mark.opsin
@pytest.mark.parametrize("offset", (0, 7, 19))
@pytest.mark.parametrize("mode", (FusionMode.GENERAL, FusionMode.AUDITED_PIN))
def test_reported_stereochemical_fusion_name_roundtrips_under_atom_permutations(offset, mode):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    graph = _reordered_reported_graph(offset)
    result = name_mol(graph, fusion_mode=mode, include_trace=True)
    assert result.ok, result.error
    assert result.parent_nomenclature == "systematic_fusion"
    assert result.name.startswith("(3S)-")
    assert "-3-methoxy-" in result.name
    assert "[1,4]thiazepino[2,3,4-ij]quinazolin-6(2H)-one" in result.name
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    plan = plan_fusion_parent(mol, atoms, mode=mode).plan
    (added_h,) = plan.derivative_state.added_hydrogen_operations
    assert added_h.locants == ("2",)
    (atom,) = added_h.atom_ids
    assert str(dict(plan.numbering.input_locant_maps[0])[atom]) == "2"
    assert mol.atoms[atom].symbol == "C" and mol.atoms[atom].total_h_count == 2
    assert not set(added_h.atom_ids).intersection(plan.derivative_state.bond_delta.hydrogenated_atom_ids)
    check = verify_with_opsin(result.name, REPORTED_SMILES, standardize_smiles=False)
    assert check.ok, check.to_dict()
    assert check.canonical_roundtrip == check.canonical_original


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", CORES)
@pytest.mark.parametrize("site", (None, 0, 1, 2))
def test_reduced_cores_and_saturated_chain_substituent_positions_roundtrip(smiles, site):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    graph = Chem.RWMol(Chem.MolFromSmiles(smiles))
    if site is not None:
        carbons = [
            atom.GetIdx()
            for atom in graph.GetAtoms()
            if atom.GetSymbol() == "C" and not atom.GetIsAromatic() and atom.GetTotalNumHs() == 2
        ]
        methyl = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(carbons[site], methyl, Chem.BondType.SINGLE)
        Chem.SanitizeMol(graph)
    result = name_mol(graph, include_trace=True)
    assert result.ok, result.error
    assert result.parent_nomenclature == "systematic_fusion"
    check = verify_with_opsin(result.name, Chem.MolToSmiles(graph), standardize_smiles=False)
    assert check.ok, check.to_dict()
    assert check.canonical_roundtrip == check.canonical_original
