"""Bridge atoms share the completed parent map with suffixes and branches."""

import random

import pytest
from rdkit import Chem

from openclatura import name_mol
from openclatura.chains import find_ring_systems
from openclatura.fusion.wrappers import plan_bridged_fusion_wrapper
from openclatura.graph_io import read_rdkit_mol
from openclatura.locants import parse_system_locant, system_locant_sort_key
from openclatura.ring_parent import RingParent

REPORTED = (
    "C=C1C(=O)[C@@]23CC[C@H]4[C@H](C(=O)O)CCC[C@@]4(C)[C@@H]2[C@@H](O)C[C@@H]1C3",
    "O=C1C[C@H]2CC[C@@H]1[C@H]1C(=O)N(c3ccc([N+](=O)[O-])cc3)C(=O)[C@@H]21",
)


def _reorder(mol, ordering):
    order = list(range(mol.GetNumAtoms()))
    if ordering == "reverse":
        order.reverse()
    elif ordering == "shuffle":
        random.Random(53).shuffle(order)
    return Chem.RenumberAtoms(mol, order)


def _assert_exact(mol):
    result = name_mol(mol, include_trace=True, verify_opsin=True)
    assert result.ok, result.error
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched", (result.name, result.opsin_check)
    back = Chem.MolFromSmiles(result.opsin_check.opsin_smiles)
    assert back is not None
    assert Chem.MolToSmiles(back) == Chem.MolToSmiles(mol), result.name
    selected = next(step for step in result.decisions if step.decision == "selected audited bridged fusion parent")
    numbered = next(step for step in result.decisions if step.decision == "selected numbering")
    assert numbered.data["atom_to_locant"] == selected.data["atom_to_locant"]
    graph = read_rdkit_mol(mol)
    ring = max(find_ring_systems(graph), key=lambda system: len(system.atoms))
    assert set(numbered.data["atom_to_locant"]) == set(ring.atoms)
    assert "ethano" in result.name
    return result


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", REPORTED)
@pytest.mark.parametrize("ordering", ["original", "reverse", "shuffle"])
def test_reported_bridge_derivatives_keep_every_atom_and_stereocenter(smiles, ordering):
    _assert_exact(_reorder(Chem.MolFromSmiles(smiles), ordering))


def _bridge_derivative(element, order):
    mol = Chem.RWMol()
    for _ in range(12):
        mol.AddAtom(Chem.Atom("C"))
    edges = set()
    for cycle in ((0, 1, 2, 3, 4, 5), (4, 5, 6, 7, 8, 9)):
        edges.update(tuple(sorted((a, b))) for a, b in zip(cycle, cycle[1:] + cycle[:1]))
    edges.update(((0, 10), (10, 11), (3, 11)))
    for left, right in sorted(edges):
        mol.AddBond(left, right, Chem.BondType.SINGLE)
    ligand = mol.AddAtom(Chem.Atom(element))
    mol.AddBond(10, ligand, Chem.BondType.SINGLE if order == 1 else Chem.BondType.DOUBLE)
    Chem.SanitizeMol(mol)
    return mol.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("element,order", [("C", 1), ("O", 1), ("O", 2), ("C", 2), ("N", 2)])
@pytest.mark.parametrize("ordering", ["original", "reverse", "shuffle"])
def test_graph_built_bridge_ligands_use_shared_numbering(element, order, ordering):
    _assert_exact(_reorder(_bridge_derivative(element, order), ordering))


@pytest.mark.parametrize("smiles", REPORTED)
def test_bridge_numbering_completes_parent_map_without_renumbering_base(smiles):
    graph = read_rdkit_mol(Chem.MolFromSmiles(smiles))
    ring = max(find_ring_systems(graph), key=lambda system: len(system.atoms))
    plan = plan_bridged_fusion_wrapper(graph, ring.atoms, mode="audited_pin")
    assert plan is not None
    mapping = plan.atom_to_locant
    assert set(mapping) == set(ring.atoms)
    assert len(set(mapping.values())) == len(mapping)
    base = dict(plan.parent.locant_maps[0])
    assert {atom: mapping[atom] for atom in base} == base
    assert RingParent.from_fusion_wrapper(plan).proof_locant_maps == (mapping,)
    next_locant = max(parse_system_locant(value).base for value in base.values()) + 1
    assert len(plan.bridges) == 1
    bridge = plan.bridges[0]
    high_endpoint = max(bridge.endpoint_atom_ids, key=lambda atom: system_locant_sort_key(base[atom]))
    first_bridge_atom = next(atom for atom in bridge.atom_ids if graph.get_bond(atom, high_endpoint) is not None)
    assert mapping[first_bridge_atom] == str(next_locant)
    assert {mapping[atom] for atom in bridge.atom_ids} == {
        str(value) for value in range(next_locant, next_locant + len(bridge.atom_ids))
    }


def test_multiple_bridges_number_higher_endpoints_before_citation_order():
    graph = read_rdkit_mol(Chem.MolFromSmiles("C12=CC=C(C=3C4=CC=C(C13)C4)O2"))
    plan = plan_bridged_fusion_wrapper(graph, graph.atoms, mode="audited_pin")
    assert plan is not None
    assert [bridge.prefix for bridge in plan.bridges] == ["epoxy", "methano"]
    assert [bridge.endpoint_locants for bridge in plan.bridges] == [("1", "4"), ("5", "8")]
    mapping = plan.atom_to_locant
    assert mapping[plan.bridges[1].atom_ids[0]] == "9"
    assert mapping[plan.bridges[0].atom_ids[0]] == "10"
    assert set(mapping) == set(graph.atoms)
    assert len(set(mapping.values())) == len(mapping)


@pytest.mark.opsin
def test_bridge_oxo_operation_not_counted_twice_as_hydrogenation():
    mol = Chem.MolFromSmiles("C12C(C=C(C3=CC=CC=C13)O2)=O")
    result = name_mol(mol, include_trace=True, verify_opsin=True)
    assert result.name == "1,4-epoxynaphthalen-2-one"
    assert result.opsin_check is not None and result.opsin_check.status == "matched"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result.opsin_check.opsin_smiles)) == Chem.MolToSmiles(mol)
    selected = next(step for step in result.decisions if step.decision == "selected audited bridged fusion parent")
    assert selected.data["derivative_operations"]["hydro"] == []


@pytest.mark.opsin
@pytest.mark.parametrize("bridge", ["O", "C=C"])
def test_self_audit_does_not_reinterpret_fusion_locants_as_von_baeyer(bridge):
    mol = Chem.MolFromSmiles(f"C12=CC=C(C3=CC=CC=C13){bridge}2")
    result = name_mol(mol, verify_self=True, verify_opsin=True)
    assert result.opsin_check is not None and result.opsin_check.status == "matched"
    assert result.self_audit is not None and result.self_audit.verdict == "abstained"
    assert "bridged fusion parent reconstruction not modelled" in result.self_audit.reason
