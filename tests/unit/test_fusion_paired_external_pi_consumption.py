"""External-pi endpoints must own an entire alternating carbon path."""

import random
from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import component_namer, name
from openclatura.audit import audit_component_reconstruction
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.external_pi_consumption import has_paired_external_pi_consumption
from openclatura.fusion.mancude import ParentBondDelta, ParentDerivativeState
from openclatura.fusion.model import BondAssignment, FusionMode
from openclatura.graph_io import read_rdkit_mol
from openclatura.name_operations import HydroOperation, OxoOperation
from openclatura.polycycle_topology import normalize_edge


def _path_state(length=3, branch=False, seed=7, charged=False, nitrogen=False):
    graph = Chem.RWMol()
    size = length + 3
    for index in range(size):
        atom = Chem.Atom("N" if nitrogen and index == 1 else "C")
        if charged and index == 1:
            atom.SetFormalCharge(1)
        graph.AddAtom(atom)
    expected = {}
    for index in range(size):
        other = (index + 1) % size
        actual_double = index < length and index % 2 == 1
        graph.AddBond(index, other, Chem.BondType.DOUBLE if actual_double else Chem.BondType.SINGLE)
        expected[normalize_edge(index, other)] = 2 if index < length and index % 2 == 0 else 1
    external = []
    for endpoint in (0, length):
        oxygen = graph.AddAtom(Chem.Atom("O"))
        graph.AddBond(endpoint, oxygen, Chem.BondType.DOUBLE)
        external.append((endpoint, oxygen))
    if branch:
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(1, carbon, Chem.BondType.SINGLE)
    graph = graph.GetMol()
    Chem.SanitizeMol(graph)
    order = list(range(graph.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    inverse = {old: new for new, old in enumerate(order)}
    mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
    atoms = frozenset(inverse[index] for index in range(size))
    assignment = BondAssignment(
        tuple(sorted((normalize_edge(inverse[u], inverse[v]), value) for (u, v), value in expected.items()))
    )
    operations = tuple(
        OxoOperation(
            key="oxo",
            reason="Graph-built external pi ligand",
            locant=str(endpoint + 1),
            parent_atom_id=inverse[endpoint],
            oxygen_atom_id=inverse[oxygen],
            bond_id=mol.get_bond(inverse[endpoint], inverse[oxygen]).idx,
        )
        for endpoint, oxygen in external
    )
    delta = ParentBondDelta(assignment, frozenset(), (), frozenset(), True)
    return mol, atoms, ParentDerivativeState(delta, oxo_operations=operations)


@pytest.mark.parametrize("length", (3, 5))
@pytest.mark.parametrize("branch", (False, True))
@pytest.mark.parametrize("seed", (7, 73, 20483))
def test_paired_consumption_preserves_internal_pi_and_hydrogen(length, branch, seed):
    assert has_paired_external_pi_consumption(*_path_state(length, branch, seed))


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_endpoint",
        "duplicate_endpoint",
        "missing_edge",
        "broken_path",
        "charge",
        "nitrogen",
        "extra_h",
        "added_h",
        "hydro",
    ),
)
def test_incomplete_or_noncarbon_paths_are_rejected(corruption):
    mol, atoms, state = _path_state(charged=corruption == "charge", nitrogen=corruption == "nitrogen")
    if corruption == "missing_endpoint":
        state = replace(state, oxo_operations=state.oxo_operations[:1])
    elif corruption == "duplicate_endpoint":
        state = replace(state, oxo_operations=(state.oxo_operations[0],) * 2)
    elif corruption in {"missing_edge", "broken_path"}:
        orders = list(state.bond_delta.assignment.orders)
        if corruption == "missing_edge":
            orders.pop()
        else:
            index = next(index for index, (_, order) in enumerate(orders) if order == 2)
            orders[index] = (orders[index][0], 1)
        state = replace(state, bond_delta=replace(state.bond_delta, assignment=BondAssignment(tuple(orders))))
    elif corruption == "extra_h":
        atom = state.oxo_operations[0].parent_atom_id
        mol.atoms[atom] = replace(mol.atoms[atom], total_h_count=mol.atoms[atom].total_h_count + 1)
    elif corruption in {"added_h", "hydro"}:
        operation = HydroOperation(
            key="added_hydrogen",
            reason="Unproved H endpoint",
            locants=("1",),
            atom_ids=(state.oxo_operations[0].parent_atom_id,),
            operation_kind="indicated_hydrogen",
        )
        if corruption == "added_h":
            state = replace(state, bond_delta=replace(state.bond_delta, added_hydrogen_operations=(operation,)))
        else:
            state = replace(state, hydro_operations=(operation,))
    assert not has_paired_external_pi_consumption(mol, atoms, state)


@pytest.fixture
def paired_fusion_parts(monkeypatch):
    captured = []
    original = component_namer.assert_component_fully_named

    def capture(mol, atoms, parts, rendered):
        if parts.parent_hydride is not None and parts.parent_hydride.uses_fusion_plan:
            captured.append((mol, set(atoms), parts))
        return original(mol, atoms, parts, rendered)

    monkeypatch.setattr(component_namer, "assert_component_fully_named", capture)
    result = name("CSc1ccc(C2c3c(oc4ccccc4c3=O)C(=O)N2c2ncccn2)cc1", verify_self=True)
    assert result.self_audit.verdict == "confirmed"
    return captured[-1]


@pytest.mark.parametrize("corruption", ("missing_endpoint", "wrong_oxo_locant", "wrong_oxo_bond", "missing_h"))
def test_paired_consumption_does_not_bypass_fusion_audit(paired_fusion_parts, corruption):
    mol, _, parts = paired_fusion_parts
    parent = parts.parent_hydride
    plan = parent.fusion_plan
    state = plan.derivative_state
    indicated = plan.indicated_hydrogens
    first, second = state.oxo_operations
    if corruption == "missing_endpoint":
        state = replace(state, oxo_operations=(first,))
    elif corruption == "wrong_oxo_locant":
        state = replace(state, oxo_operations=(replace(first, locant="1"), second))
    elif corruption == "wrong_oxo_bond":
        state = replace(state, oxo_operations=(replace(first, bond_id=-1), second))
    else:
        indicated = ()
    audit = audit_fusion_plan(
        mol,
        parent.atoms,
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=indicated,
        derivative_state=state,
        mode=FusionMode.AUDITED_PIN,
    )
    assert not audit.confirmed


@pytest.mark.parametrize("corruption", ("missing_oxo", "moved_oxo", "added_h"))
def test_paired_consumption_reconstruction_still_rejects_corrupt_operations(paired_fusion_parts, corruption):
    mol, atoms, parts = paired_fusion_parts
    assert audit_component_reconstruction(mol, parts, atoms).verdict == "confirmed"
    bad = deepcopy(parts)
    if corruption == "missing_oxo":
        bad.principal_group = replace(bad.principal_group, locants=bad.principal_group.locants[:1])
    elif corruption == "moved_oxo":
        bad.principal_group = replace(bad.principal_group, locants=("4", "9"))
    else:
        bad.indicated_hydrogens.append("4")
    assert audit_component_reconstruction(mol, bad, atoms).verdict != "confirmed"
