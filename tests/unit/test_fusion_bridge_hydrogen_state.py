"""Bridge composition preserves the selected fusion parent's hydrogen proof."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.fusion import wrappers
from openclatura.fusion.model import FusionMode
from openclatura.graph_io import read_rdkit_mol


def _bridged_furo_graph(heteroatom):
    graph = Chem.RWMol()
    for symbol in ("C", "C", "O", "C", "C", "C", heteroatom, "C", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    for left, right in (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (3, 4),
        (4, 5),
        (5, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 4),
    ):
        graph.AddBond(left, right, Chem.BondType.DOUBLE if (left, right) in {(4, 5), (7, 8)} else Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.parametrize("heteroatom,parent", [("N", "1H-furo[3,4-b]pyrrole"), ("O", "furo[3,4-b]furan")])
@pytest.mark.parametrize("mode", ["general", "audited_pin"])
@pytest.mark.parametrize("reverse", [False, True])
def test_bridge_keeps_fusion_pi_redistribution_and_exact_hydrogenation(heteroatom, parent, mode, reverse):
    graph = _bridged_furo_graph(heteroatom)
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    mol = read_rdkit_mol(graph)
    plan = wrappers.plan_bridged_fusion_wrapper(mol, mol.atoms, mode=mode)
    assert plan is not None
    state = plan.derivative_state
    assert state is plan.parent.fusion_plan.derivative_state
    assert tuple(operation.locants for operation in state.hydro_operations) == (("4", "6"),)
    assert state.pi_redistribution is not None
    assert len(state.pi_redistribution.removed_bond_ids) == 2
    assert len(state.pi_redistribution.added_bond_ids) == 1
    assert state.pi_redistribution.hydrogenated_atom_ids == set(plan.bridges[0].endpoint_atom_ids)
    assert set(plan.atom_to_locant) == set(mol.atoms)
    assert not state.unsaturation_operations
    result = name_mol(graph, fusion_mode=mode, verify_opsin=opsin_available())
    assert (
        result.name == f"4,6-dihydro-4,6-methano-{parent}"
        if heteroatom == "N"
        else result.name == f"4,6-dihydro-4,6-methano{parent}"
    )
    assert result.parent_nomenclature == "bridged_fusion"
    if opsin_available():
        assert result.opsin_check.status == "matched"
        assert Chem.MolToSmiles(Chem.MolFromSmiles(result.opsin_check.opsin_smiles)) == Chem.MolToSmiles(graph)


@pytest.mark.parametrize("mutation", ["retained_context", "hydro", "pi_certificate"])
def test_bridge_audit_rejects_lost_or_corrupted_fusion_hydrogen_state(mutation):
    mol = read_rdkit_mol(_bridged_furo_graph("N"))
    plan = wrappers.plan_bridged_fusion_wrapper(mol, mol.atoms, mode="general")
    assert plan is not None
    state = plan.derivative_state
    if mutation == "hydro":
        state = replace(state, hydro_operations=())
    elif mutation == "pi_certificate":
        state = replace(state, pi_redistribution=replace(state.pi_redistribution, hydrogenated_atom_ids=frozenset()))
    assert (
        wrappers._audit_bridge_plan(
            mol,
            frozenset(mol.atoms),
            plan.parent.atom_ids,
            dict(plan.parent.locant_maps[0]),
            plan.bridges,
            plan.parent.selected_bond_model,
            state,
            fusion_plan=None if mutation == "retained_context" else plan.parent.fusion_plan,
        )
        is None
    )


@pytest.mark.parametrize("smiles", ["C1C2NC1C1=NNC=C21", "C1C2NC1C1=NNN=C21"])
@pytest.mark.parametrize("mode", ["general", "audited_pin"])
@pytest.mark.parametrize("reverse", [False, True])
def test_competing_epimino_and_methano_parents_preserve_exact_hydrogen_state(smiles, mode, reverse):
    graph = Chem.MolFromSmiles(smiles)
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    mol = read_rdkit_mol(graph)
    bridge_n = next(atom for atom, value in mol.atoms.items() if value.symbol == "N" and not value.is_aromatic)
    epimino_parent = wrappers._systematic_fusion_parent(mol, frozenset(mol.atoms) - {bridge_n}, FusionMode(mode))
    assert epimino_parent is not None
    assert tuple(op.locants for op in epimino_parent.fusion_plan.derivative_state.hydro_operations) == (("5", "6"),)
    result = name_mol(graph, fusion_mode=mode, verify_opsin=opsin_available())
    assert result.parent_nomenclature == "bridged_fusion"
    assert result.name
    if opsin_available():
        assert result.opsin_check.status == "matched", (result.name, result.opsin_check)
        assert Chem.MolToSmiles(Chem.MolFromSmiles(result.opsin_check.opsin_smiles)) == Chem.MolToSmiles(graph)
