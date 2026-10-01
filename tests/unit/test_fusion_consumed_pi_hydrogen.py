"""External pi consumption can require H even with no residual hydro pairs."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import _audit_derivative_state
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol

COMPONENTS = (
    pytest.param(("N", "C", "O"), (5, 6), id="24912-oxazole"),
    pytest.param(("O", "C", "N"), (6, 7), id="24914-oxazole"),
    pytest.param(("O", "N", "C"), (6, 7), id="129112-isoxazole"),
    pytest.param(("O", "N", "N"), (6, 7), id="129113-oxadiazole"),
)


def _external_pi_derivative(component, component_double, external, side_length, reverse):
    graph = Chem.RWMol()
    for symbol in ("C", "C", "C", "C", "O", *component, external):
        graph.AddAtom(Chem.Atom(symbol))
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (3, 5), (5, 6), (6, 7), (7, 2), (0, 8))
    for edge in edges:
        graph.AddBond(
            *edge, Chem.BondType.DOUBLE if edge in {(2, 3), component_double, (0, 8)} else Chem.BondType.SINGLE
        )
    previous = 1
    for _ in range(side_length):
        atom = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(previous, atom, Chem.BondType.SINGLE)
        previous = atom
    source = graph.GetMol()
    Chem.SanitizeMol(source)
    order = list(range(source.GetNumAtoms()))
    if reverse:
        order.reverse()
        source = Chem.RenumberAtoms(source, order)
    return source, order


@pytest.mark.parametrize("component,component_double", COMPONENTS)
@pytest.mark.parametrize("external", ("O", "N", "C"))
@pytest.mark.parametrize("side_length", (0, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_consumed_parent_pi_pair_retains_partner_hydrogen(component, component_double, external, side_length, reverse):
    source, order = _external_pi_derivative(component, component_double, external, side_length, reverse)
    mol = read_rdkit_mol(source)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    planned = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(planned, FusionConfirmed), planned
    plan = planned.plan
    state = plan.derivative_state
    assert not state.hydro_operations
    assert not state.unsaturation_operations
    (added,) = state.added_hydrogen_operations
    assert added.atom_ids == (order.index(1),)
    locants = dict(plan.numbering.input_locant_maps[0])
    assert added.locants == (str(locants[order.index(1)]),)
    assert set(added.bond_ids) == {mol.get_bond(order.index(1), order.index(other)).idx for other in (0, 2)}
    (operation,) = state.external_pi_operations
    assert operation.parent_atom_id == order.index(0)
    assert operation.external_atom_id == order.index(8)
    assert not set(added.atom_ids).intersection((operation.parent_atom_id, operation.external_atom_id))
    named = name_mol(source, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(source), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip


@pytest.mark.parametrize("corruption", ("missing", "atom_ids", "bond_ids", "locants"))
def test_consumed_pi_composition_audit_rejects_lost_or_corrupt_hydrogen(corruption):
    source, _ = _external_pi_derivative(("O", "N", "N"), (6, 7), "N", 0, False)
    mol = read_rdkit_mol(source)
    atoms = frozenset(range(8))
    plan = plan_fusion_parent(mol, atoms, mode="audited_pin").plan
    state = plan.derivative_state
    (added,) = state.added_hydrogen_operations
    changed = () if corruption == "missing" else (replace(added, **{corruption: ()}),)
    corrupted = replace(state, bond_delta=replace(state.bond_delta, added_hydrogen_operations=changed))
    errors = []
    _audit_derivative_state(mol, atoms, plan.numbering, plan.bond_model, plan.indicated_hydrogens, corrupted, errors)
    assert errors
