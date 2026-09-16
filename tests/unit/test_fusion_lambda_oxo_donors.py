"""Neutral lambda-oxo spectators preserve fusion donor and carbon-H proofs."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import _has_carbon_h_oxo_bond_consumption, audit_fusion_plan
from openclatura.fusion.exocyclic import is_neutral_external_pi_ligand, neutral_lambda_oxo_bonding_number
from openclatura.fusion.indicated_hydrogen import aromatic_nitrogen_lone_pair_sites
from openclatura.fusion.mancude import parent_derivative_state, prove_pi_redistribution
from openclatura.fusion.model import FusionConfirmed, FusionGraph, FusionGraphAtom, FusionGraphBond, FusionMode
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.molecule import Molecule

CASES = (
    (
        3589,
        "COc1ccc(-n2nc3c(c2NC(=O)Cc2ccccc2)CS(=O)C3)cc1",
        "N-(2-(4-methoxyphenyl)-5-oxo-4H,6H-5lambda^4-thieno[3,4-c]pyrazol-3-yl)-2-phenylacetamide",
        ("4", "6"),
        4,
    ),
    (
        3924,
        "Cn1nc(C(=O)NCCN(Cc2ccco2)C2CCCC2)c2c1-c1ccccc1S(=O)(=O)C2",
        "N-(2-(((furan-2-yl)methyl)(cyclopentyl)amino)ethyl)-1-methyl-5,5-dioxo-4H-"
        "5lambda^6-benzo[1',2':2,3]thiino[4,5-c]pyrazole-3-carboxamide",
        ("4",),
        6,
    ),
)


def _plan(graph, mode="general"):
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=mode)
    assert isinstance(result, FusionConfirmed), result
    return mol, atoms, result.plan


def _audit(mol, atoms, plan, **changes):
    arguments = dict(
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        derivative_state=plan.derivative_state,
        charge_operations=plan.charge_operations,
        lambda_descriptors=plan.lambda_descriptors,
    )
    arguments.update(changes)
    return audit_fusion_plan(mol, atoms, **arguments)


def _exact_opsin(name, graph):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required for exact round-trip verification")
    check = verify_with_opsin(name, Chem.MolToSmiles(graph), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip


@pytest.mark.opsin
@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case[0]))
@pytest.mark.parametrize("seed", [None, 0, 17])
@pytest.mark.parametrize("mode", ["general", "audited_pin"])
def test_reported_lambda_oxo_families_keep_exact_fusion_and_local_roles(case, seed, mode):
    _, smiles, expected, hydrogen, bonding_number = case
    graph = Chem.MolFromSmiles(smiles)
    order = list(range(graph.GetNumAtoms()))
    if seed is None:
        order.reverse()
    else:
        random.Random(seed).shuffle(order)
    graph = Chem.RenumberAtoms(graph, order)
    mol, atoms, plan = _plan(graph, mode)
    assert _audit(mol, atoms, plan).confirmed
    assert tuple(map(str, plan.indicated_hydrogens)) == hydrogen
    donors = aromatic_nitrogen_lone_pair_sites(mol, plan.abstract_parent_graph)
    assert len(donors) == 1
    sulfur = next(atom for atom in atoms if mol.atoms[atom].symbol == "S")
    assert neutral_lambda_oxo_bonding_number(mol, sulfur) == bonding_number
    assert [(item.atom_id, item.bonding_number) for item in plan.lambda_descriptors] == [(sulfur, bonding_number)]
    assert all(
        bond_order == 1
        for assignment in plan.bond_model.allowed_kekule_assignments
        for edge, bond_order in assignment.orders
        if (donors | {sulfur}).intersection(edge)
    )
    state = plan.derivative_state
    assert not state.hydro_operations
    assert not state.unsaturation_operations
    assert not state.added_hydrogen_operations
    assert len(state.oxo_operations) == (bonding_number - 2) // 2
    assert {operation.parent_atom_id for operation in state.oxo_operations} == {sulfur}
    result = name_mol(graph, fusion_mode=mode, include_trace=True)
    assert result.error is None
    assert result.name == expected
    # The amide-rooted 3589 parent is not itself the fused substituent.
    if case[0] == 3924:
        assert result.parent_nomenclature == "systematic_fusion"
    _exact_opsin(result.name, graph)


def _graph_built_oxo_pyrazole(benzo, oxo_count, alkyl_length, chalcogen="S"):
    graph = Chem.RWMol()
    symbols = (
        ("N", "N", "C", "C", "C", "C", "C", "S", "C", "C", "C", "C", "C")
        if benzo
        else ("N", "N", "C", "C", "C", "C", "S", "C")
    )
    for symbol in symbols:
        graph.AddAtom(Chem.Atom(chalcogen if symbol == "S" else symbol))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    doubles = {(1, 2), (3, 4)}
    if benzo:
        sulfur = 7
        edges.extend([(4, 5), (5, 6), (6, 7), (7, 8), (8, 3), (6, 9), (9, 10), (10, 11), (11, 12), (12, 5)])
        doubles.update({(5, 6), (9, 10), (11, 12)})
    else:
        sulfur = 6
        edges.extend([(4, 5), (5, 6), (6, 7), (7, 3)])
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in doubles else Chem.BondType.SINGLE)
    for _ in range(oxo_count):
        oxygen = graph.AddAtom(Chem.Atom("O"))
        graph.AddBond(sulfur, oxygen, Chem.BondType.DOUBLE)
    last = 0
    for _ in range(alkyl_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("benzo", [False, True])
@pytest.mark.parametrize("oxo_count", [1, 2])
@pytest.mark.parametrize("alkyl_length", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("chalcogen", ["S", "Se"])
def test_graph_built_oxidation_and_donor_ligand_variants(benzo, oxo_count, alkyl_length, reverse, chalcogen):
    graph = _graph_built_oxo_pyrazole(benzo, oxo_count, alkyl_length, chalcogen)
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    mol, atoms, plan = _plan(graph)
    assert _audit(mol, atoms, plan).confirmed
    assert aromatic_nitrogen_lone_pair_sites(mol, plan.abstract_parent_graph)
    assert len(plan.derivative_state.oxo_operations) == oxo_count
    if benzo:
        assert not plan.derivative_state.hydro_operations
    else:
        assert plan.derivative_state.pi_redistribution is not None
        assert tuple(
            locant for operation in plan.derivative_state.hydro_operations for locant in operation.locants
        ) == ("4", "6")
    result = name_mol(graph, fusion_mode="general", include_trace=True)
    assert result.error is None
    assert result.parent_nomenclature == "systematic_fusion"
    _exact_opsin(result.name, graph)


@pytest.mark.parametrize("corruption", ["donor", "hydrogen", "oxo", "lambda"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case[0]))
def test_independent_audit_rejects_missing_lambda_oxo_donor_proof(case, corruption):
    mol, atoms, plan = _plan(Chem.MolFromSmiles(case[1]))
    if corruption == "donor":
        changed = dict(bond_model=parent_bond_model(plan.abstract_parent_graph))
    elif corruption == "hydrogen":
        changed = dict(indicated_hydrogens=())
    elif corruption == "oxo":
        changed = dict(derivative_state=replace(plan.derivative_state, oxo_operations=()))
    else:
        changed = dict(lambda_descriptors=())
    assert not _audit(mol, atoms, plan, **changed).confirmed


@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case[0]))
@pytest.mark.parametrize(
    "corruption",
    ["extra_sigma", "extra_oxo", "center_charge", "ligand_charge", "missing", "duplicate", "wrong_ligand", "parent_pi"],
)
def test_pin_spectator_gate_rejects_unproved_ligands_charge_and_oxo_ownership(case, corruption):
    mol, atoms, plan = _plan(Chem.MolFromSmiles(case[1]), "audited_pin")
    state = plan.derivative_state
    assert _has_carbon_h_oxo_bond_consumption(mol, atoms, state)
    sulfur = state.oxo_operations[0].parent_atom_id
    oxygen = state.oxo_operations[0].oxygen_atom_id
    if corruption in {"extra_sigma", "extra_oxo"}:
        ligand = mol.add_atom("C" if corruption == "extra_sigma" else "O", idx=99)
        mol.add_bond(sulfur, ligand.idx, order=1 if corruption == "extra_sigma" else 2)
    elif corruption == "center_charge":
        mol.update_atom(sulfur, charge=1)
    elif corruption == "ligand_charge":
        mol.update_atom(oxygen, charge=-1)
    elif corruption == "missing":
        state = replace(state, oxo_operations=state.oxo_operations[1:])
    elif corruption == "duplicate":
        state = replace(state, oxo_operations=(*state.oxo_operations, state.oxo_operations[0]))
    elif corruption == "wrong_ligand":
        state = replace(
            state, oxo_operations=(replace(state.oxo_operations[0], oxygen_atom_id=sulfur), *state.oxo_operations[1:])
        )
    else:
        changed = replace(
            state.bond_delta.assignment,
            orders=tuple((edge, 2 if sulfur in edge else order) for edge, order in state.bond_delta.assignment.orders),
        )
        state = replace(state, bond_delta=replace(state.bond_delta, assignment=changed))
    assert not _has_carbon_h_oxo_bond_consumption(mol, atoms, state)
    assert not _audit(mol, atoms, plan, mode=FusionMode.AUDITED_PIN, derivative_state=state).confirmed


@pytest.mark.parametrize(
    "corruption",
    [
        "s_charge",
        "s_h",
        "s_aromatic",
        "o_charge",
        "o_h",
        "o_branch",
        "skeletal_pi",
        "extra_sigma",
        "imino",
        "tellurium",
    ],
)
def test_lambda_oxo_role_rejects_unproved_local_valence(corruption):
    graph = _graph_built_oxo_pyrazole(False, 1, 1)
    mol, _, plan = _plan(graph)
    sulfur = next(atom.idx for atom in mol.atoms.values() if atom.symbol == "S")
    oxygen = next(other for other in mol.get_neighbors(sulfur) if mol.atoms[other].symbol == "O")
    assert neutral_lambda_oxo_bonding_number(mol, sulfur) == 4
    if corruption == "s_charge":
        mol.update_atom(sulfur, charge=1)
    elif corruption == "s_h":
        mol.update_atom(sulfur, total_h_count=1)
    elif corruption == "s_aromatic":
        mol.update_atom(sulfur, is_aromatic=True)
    elif corruption == "o_charge":
        mol.update_atom(oxygen, charge=-1)
    elif corruption == "o_h":
        mol.update_atom(oxygen, total_h_count=1)
    elif corruption in {"o_branch", "extra_sigma"}:
        mol.add_atom("C", idx=99)
        mol.add_bond(oxygen if corruption == "o_branch" else sulfur, 99)
    elif corruption == "skeletal_pi":
        other = next(other for other in mol.get_neighbors(sulfur) if other != oxygen)
        mol.update_bond(mol.get_bond(sulfur, other).idx, order=2)
    elif corruption == "imino":
        mol.update_atom(oxygen, symbol="N", total_h_count=1)
    else:
        mol.update_atom(sulfur, symbol="Te")
    assert neutral_lambda_oxo_bonding_number(mol, sulfur) is None
    assert not is_neutral_external_pi_ligand(mol, sulfur, oxygen)
    assert not aromatic_nitrogen_lone_pair_sites(mol, plan.abstract_parent_graph)


def _redistribution_case(oxo_count, chalcogen="S"):
    mol = Molecule()
    for atom in range(7):
        mol.add_atom(chalcogen if atom == 6 else "C", idx=atom)
    for atom in range(7):
        mol.add_bond(atom, (atom + 1) % 7, order=2 if atom in {1, 4} else 1)
    for atom, value in tuple(mol.atoms.items()):
        load = sum(mol.get_bond(atom, other).order for other in mol.get_neighbors(atom))
        mol.update_atom(atom, total_h_count=value.element.standard_valence - load)
    atoms = frozenset(mol.atoms)
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, value.symbol, forced_single=atom == 6) for atom, value in mol.atoms.items()),
        bonds=tuple(FusionGraphBond(tuple(sorted((bond.u, bond.v))), "mancude") for bond in mol.bonds.values()),
    )
    model = parent_bond_model(graph)
    for i in range(oxo_count):
        mol.add_atom("O", idx=10 + i)
        mol.add_bond(6, 10 + i, order=2)
    state = parent_derivative_state(mol, atoms, model, {atom: str(atom + 1) for atom in atoms})
    return mol, atoms, model, state


@pytest.mark.parametrize("oxo_count", [1, 2])
@pytest.mark.parametrize("chalcogen", ["S", "Se"])
def test_lambda_oxo_is_an_unchanged_spectator_of_proved_pi_redistribution(oxo_count, chalcogen):
    mol, atoms, model, state = _redistribution_case(oxo_count, chalcogen)
    proof = state.pi_redistribution
    assert proof is not None
    assert proof.hydrogenated_atom_ids == {0, 3}
    assert set(proof.final_assignment.orders) == {
        (tuple(sorted((bond.u, bond.v))), bond.order)
        for bond in mol.bonds.values()
        if bond.u in atoms and bond.v in atoms
    }
    assert all(order == 1 for edge, order in proof.final_assignment.orders if 6 in edge)
    assert len(state.oxo_operations) == oxo_count
    assert prove_pi_redistribution(mol, atoms, model, state.bond_delta, oxo_operations=state.oxo_operations) == proof
    assert prove_pi_redistribution(mol, atoms, model, state.bond_delta) is None


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "wrong_bond", "charge", "hydrogen", "parent_pi"])
def test_sulfur_redistribution_requires_complete_typed_oxo_and_single_skeleton(corruption):
    mol, atoms, model, state = _redistribution_case(2)
    operations = state.oxo_operations
    assert state.pi_redistribution is not None
    if corruption == "missing":
        operations = operations[:1]
    elif corruption == "duplicate":
        operations = (operations[0], operations[0])
    elif corruption == "wrong_bond":
        operations = (replace(operations[0], bond_id=mol.get_bond(0, 1).idx), operations[1])
    elif corruption == "charge":
        mol.update_atom(6, charge=1)
    elif corruption == "hydrogen":
        mol.update_atom(6, total_h_count=1)
    else:
        mol.update_bond(mol.get_bond(6, 0).idx, order=2)
    assert prove_pi_redistribution(mol, atoms, model, state.bond_delta, oxo_operations=operations) is None
