"""Cited nitrogen H must precede the completed carbon-H matching proof."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.fusion import indicated_hydrogen as intrinsic
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import AuditStatus, FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles


@pytest.mark.parametrize("mode", ["general", "audited_pin"])
@pytest.mark.parametrize("reverse", [False, True])
def test_saturated_nitrogen_h_does_not_hide_intrinsic_carbon_h(mode, reverse):
    graph = Chem.MolFromSmiles("C1CC2N=CNC2=N1")
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    result = name_mol(graph, fusion_mode=mode, verify_opsin=opsin_available())
    assert result.parent_nomenclature == "systematic_fusion"
    assert result.name == "6,6a-dihydro-3H,5H-pyrrolo[2,3-d]imidazole"
    if opsin_available():
        assert result.opsin_check.status == "matched"
        assert Chem.MolToSmiles(Chem.MolFromSmiles(result.opsin_check.opsin_smiles)) == Chem.MolToSmiles(graph)


@pytest.mark.parametrize("mutation", ["omit_n", "omit_c", "hydro_c"])
def test_mixed_cited_nh_carbon_h_audit_rejects_incomplete_or_overlapping_citations(mutation):
    mol = read_smiles("C1CC2N=CNC2=N1")
    result = plan_fusion_parent(mol, mol.atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    atom_by_locant = {locant: atom for atom, locant in plan.numbering.input_locant_maps[0]}
    citations = tuple(
        locant
        for locant in plan.indicated_hydrogens
        if not (
            (mutation == "omit_n" and mol.atoms[atom_by_locant[locant]].symbol == "N")
            or (mutation == "omit_c" and mol.atoms[atom_by_locant[locant]].symbol == "C")
        )
    )
    if mutation == "hydro_c":
        hydro_atoms = {atom for op in plan.derivative_state.hydro_operations for atom in op.atom_ids}
        citations += tuple(locant for locant, atom in atom_by_locant.items() if atom in hydro_atoms)
    audit = audit_fusion_plan(
        mol,
        frozenset(mol.atoms),
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=citations,
        derivative_state=plan.derivative_state,
        mode="audited_pin",
    )
    assert audit.status is AuditStatus.MISMATCH


def test_complete_saturated_nh_composition_keeps_hydro_model_without_carbon_tautomer(monkeypatch):
    mol = read_smiles("N1CCCC2C1CNC2")
    result = plan_fusion_parent(mol, mol.atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    nitrogen = frozenset(atom for atom, value in mol.atoms.items() if value.symbol == "N")
    carbon = frozenset(mol.atoms) - nitrogen
    locants = dict(plan.numbering.input_locant_maps[0])
    assert {locants[atom] for atom in nitrogen} == set(plan.indicated_hydrogens)
    assert len(nitrogen) == 2

    def unexpected(*args, **kwargs):
        pytest.fail("proved complete hydrogenation must not search for a competing carbon tautomer")

    monkeypatch.setattr(intrinsic, "_nitrogen_composition_parent_model", unexpected)
    model, sites = intrinsic.intrinsic_carbon_parent_model(
        mol,
        plan.abstract_parent_graph,
        plan.bond_model,
        locants,
        carbon,
        cited_nitrogen_hydrogen_atom_ids=nitrogen,
    )
    assert model is plan.bond_model
    assert not sites
