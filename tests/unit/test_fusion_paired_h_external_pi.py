"""Joint indicated-H/external-pi operations must survive parent composition."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion import mancude
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _fused_nitrogen_derivative(alkyl_length, charge, external_symbol):
    editable = Chem.RWMol()
    for index in range(10):
        atom = Chem.Atom("N" if index in {0, 7} else "C")
        if index == 0:
            atom.SetFormalCharge(charge)
            atom.SetNumExplicitHs(charge)
        if index == 7:
            atom.SetNumExplicitHs(1)
        editable.AddAtom(atom)
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 9), (9, 0), (3, 8), (8, 7), (7, 6), (6, 5), (5, 4))
    for u, v in edges:
        editable.AddBond(u, v, Chem.BondType.DOUBLE if (u, v) in {(3, 4), (6, 5)} else Chem.BondType.SINGLE)
    external = editable.AddAtom(Chem.Atom(external_symbol))
    editable.AddBond(8, external, Chem.BondType.DOUBLE)
    previous = 0
    for _ in range(alkyl_length):
        atom = editable.AddAtom(Chem.Atom("C"))
        editable.AddBond(previous, atom, Chem.BondType.SINGLE)
        previous = atom
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("alkyl_length", (1, 2, 3))
@pytest.mark.parametrize("charge", (0, 1))
@pytest.mark.parametrize("external_symbol", ("O", "N"))
@pytest.mark.parametrize("reverse", (False, True))
def test_paired_indicated_h_external_pi_preserves_fusion(alkyl_length, charge, external_symbol, reverse):
    source = _fused_nitrogen_derivative(alkyl_length, charge, external_symbol)
    if reverse:
        source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    graph = read_rdkit_mol(source)
    atoms = max(find_ring_systems(graph), key=lambda system: len(system.atoms)).atoms
    planned = plan_fusion_parent(graph, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(planned, FusionConfirmed)
    plan = planned.plan
    state = plan.derivative_state
    assert len(state.external_pi_operations) == 1
    indicated = {atom for atom, locant in plan.numbering.input_locant_maps[0] if locant in plan.indicated_hydrogens}
    assert not indicated.intersection(state.bond_delta.hydrogenated_atom_ids)
    assert not state.added_hydrogen_operations
    result = name_mol(source)
    assert result.error is None
    assert any(parent in result.name for parent in ("pyrido[4,3-c]pyridin", "2,6-naphthyridin"))
    if opsin_available():
        check = verify_with_opsin(result.name, Chem.MolToSmiles(source), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip


def test_unpaired_orientation_still_requires_external_pi_recomposition(monkeypatch):
    graph = read_rdkit_mol(_fused_nitrogen_derivative(1, 1, "O"))
    atoms = frozenset(range(10))
    original = mancude._external_pi_parent_delta
    recomposed_oxo_locants = []

    def record(mol, parent_atoms, model, external_atoms, locants):
        recomposed_oxo_locants.extend(str(locants[atom]) for atom in external_atoms)
        return original(mol, parent_atoms, model, external_atoms, locants)

    monkeypatch.setattr(mancude, "_external_pi_parent_delta", record)
    planned = plan_fusion_parent(graph, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(planned, FusionConfirmed)
    assert recomposed_oxo_locants and set(recomposed_oxo_locants) == {"1"}
    assert {state.derivative_state.oxo_operations[0].locant for state in planned.plan.numbering_variants} == {"5"}
    state = planned.plan.derivative_state
    indicated = {
        atom
        for atom, locant in planned.plan.numbering.input_locant_maps[0]
        if locant in planned.plan.indicated_hydrogens
    }
    oxo = state.oxo_operations[0].parent_atom_id
    assert any(
        order == 2 and oxo in edge and indicated.intersection(edge)
        for edge, order in state.bond_delta.assignment.orders
    )
