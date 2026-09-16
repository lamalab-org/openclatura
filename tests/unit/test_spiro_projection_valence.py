"""Spiro-side projection preserves the isolated hydride's valence proof."""

import random
from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name, name_mol, opsin_available, verify_with_opsin
from openclatura.graph_io import read_rdkit_mol
from openclatura.spiro_subgraph import _project_spiro_side_molecule, plan_graph_spiro_side

REPORT_SMILES = "O=C(OC(=O)C(F)(F)F)C(Cc1cscn1)CN1C2CCC1CC1(C2)OCc2ccc(F)cc21"
REPORT_NAME = (
    "2,2,2-trifluoroacetyl 2-((5'-fluorospiro[8-azabicyclo[3.2.1]octane-3,3'-"
    "(1,3-dihydrobenzo[c]furan)]-8-yl)methyl)-3-(1,3-thiazol-4-yl)propanoate"
)


def _graph(*, oxygen_position=1, substitution=None):
    graph = Chem.RWMol()
    for index in range(16):
        graph.AddAtom(Chem.Atom("O" if index == oxygen_position else "N" if index == 15 else "C"))
    for a, b in ((0, 1), (1, 2), (2, 3), (8, 0)):
        graph.AddBond(a, b, Chem.BondType.SINGLE)
    benzene = list(range(3, 9))
    for offset, a in enumerate(benzene):
        graph.AddBond(a, benzene[(offset + 1) % 6], Chem.BondType.DOUBLE if offset % 2 == 0 else Chem.BondType.SINGLE)
    for a, b in ((9, 11), (11, 0), (0, 12), (12, 10), (9, 13), (13, 14), (14, 10), (9, 15), (15, 10)):
        graph.AddBond(a, b, Chem.BondType.SINGLE)
    # An acid side chain makes the fused ring a cited spiro side, as in 1219.
    chain = [graph.AddAtom(Chem.Atom(symbol)) for symbol in ("C", "C", "O", "O")]
    for a, b, order in ((15, chain[0], 1), (chain[0], chain[1], 1), (chain[1], chain[2], 2), (chain[1], chain[3], 1)):
        graph.AddBond(a, b, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    side_atoms = set(range(9))
    if substitution is not None:
        parent, symbol = substitution
        added = graph.AddAtom(Chem.Atom(symbol))
        graph.AddBond(parent, added, Chem.BondType.SINGLE)
        side_atoms.add(added)
    Chem.SanitizeMol(graph)
    return graph.GetMol(), side_atoms


def _orders(graph):
    forward = list(range(graph.GetNumAtoms()))
    shuffled = forward.copy()
    random.Random(1219).shuffle(shuffled)
    return forward, forward[::-1], shuffled


def _snapshot(mol):
    return deepcopy((mol.atoms, mol.bonds, mol._adj, mol._bond_lookup, mol.accurate_cip, mol.substituted_symbols))


def test_projection_caps_only_junction_and_preserves_original_graph():
    graph, side_atoms = _graph(substitution=(5, "F"))
    mol = read_rdkit_mol(graph)
    original = _snapshot(mol)
    projected = _project_spiro_side_molecule(mol, side_atoms, 0)
    assert _snapshot(mol) == original
    assert mol.atoms[0].total_h_count == 0
    assert mol.subgraph(side_atoms).atoms[0].total_h_count == 0
    assert projected.atoms[0] == replace(mol.atoms[0], total_h_count=2)
    assert projected.atoms[0].explicit_h_count == 0
    assert set(projected.atoms) == side_atoms
    assert projected.bonds == {i: b for i, b in mol.bonds.items() if b.u in side_atoms and b.v in side_atoms}
    for atom in projected:
        assert (
            atom.total_h_count + sum(projected.get_bond(atom.idx, n).order for n in projected.get_neighbors(atom.idx))
            == atom.element.standard_valence
        )
        if atom.idx != 0:
            assert atom == mol.atoms[atom.idx]


@pytest.mark.parametrize(
    "change",
    [
        "charged",
        "heteroatom",
        "aromatic",
        "hydrogen",
        "explicit_hydrogen",
        "multiple_bond",
        "one_external",
        "no_external",
    ],
)
def test_projection_does_not_cap_outside_neutral_carbon_spiro_scope(change):
    graph, side_atoms = _graph()
    mol = read_rdkit_mol(graph)
    changes = {
        "charged": {"charge": 1},
        "heteroatom": {"symbol": "N"},
        "aromatic": {"is_aromatic": True},
        "hydrogen": {"total_h_count": 1},
        "explicit_hydrogen": {"explicit_h_count": 1},
    }
    if change in changes:
        mol.atoms[0] = replace(mol.atoms[0], **changes[change])
    elif change == "multiple_bond":
        bond = mol.get_bond(0, 11)
        mol.bonds[bond.idx] = replace(bond, order=2)
    elif change == "one_external":
        side_atoms.add(11)
    else:
        side_atoms = set(mol.atoms)
    original = _snapshot(mol)
    projected = _project_spiro_side_molecule(mol, side_atoms, 0)
    assert projected.atoms[0] == mol.atoms[0]
    assert _snapshot(mol) == original


CASES = [(oxygen, substitution) for oxygen in (1, 2) for substitution in (None, (5, "F"), (3 - oxygen, "C"))]


@pytest.mark.parametrize("oxygen,substitution", CASES)
def test_graph_built_fused_side_proves_net_hydrogenation(oxygen, substitution):
    graph, side_atoms = _graph(oxygen_position=oxygen, substitution=substitution)
    names = []
    for order in _orders(graph):
        mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
        original = _snapshot(mol)
        junction = order.index(0)
        side = plan_graph_spiro_side(
            mol,
            {order.index(a) for a in side_atoms},
            junction,
            required_core={order.index(a) for a in range(9)},
        )
        assert side is not None
        assert _snapshot(mol) == original
        state = side.side_parts.parent_hydride.derivative_state
        assert state is not None
        assert {a for op in state.hydro_operations for a in op.atom_ids} == {junction, order.index(3 - oxygen)}
        assert not any(mol.atoms[a].is_aromatic for op in state.hydro_operations for a in op.atom_ids)
        assert side.side_parts.parent_atom_ids_by_locant[side.side_locant] == junction
        names.append(side.side_parent_name)
    assert len(set(names)) == 1


def test_report1219_uses_dihydro_side():
    result = name(REPORT_SMILES, verify_opsin=False)
    assert result, result
    assert result.name == REPORT_NAME


@pytest.mark.opsin
def test_report1219_exact_opsin_and_atom_permutations():
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = Chem.MolFromSmiles(REPORT_SMILES)
    for order in _orders(graph):
        result = name_mol(Chem.RenumberAtoms(graph, order), verify_opsin=False)
        assert result, result
        assert "1,3-dihydrobenzo[c]furan" in result.name
        assert "tetrahydro" not in result.name
        check = verify_with_opsin(result.name, REPORT_SMILES, standardize_smiles=False)
        assert check.ok, check.to_dict()
        assert check.canonical_roundtrip == Chem.MolToSmiles(graph), check.to_dict()


@pytest.mark.opsin
@pytest.mark.parametrize("oxygen,substitution", CASES)
def test_graph_built_spiro_exact_opsin_and_atom_permutations(oxygen, substitution):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph, _ = _graph(oxygen_position=oxygen, substitution=substitution)
    original = Chem.MolToSmiles(graph)
    names = []
    for order in _orders(graph):
        result = name_mol(Chem.RenumberAtoms(graph, order), verify_opsin=False)
        assert result, result
        check = verify_with_opsin(result.name, original, standardize_smiles=False)
        assert check.ok, check.to_dict()
        assert check.canonical_roundtrip == original, check.to_dict()
        names.append(result.name)
    assert len(set(names)) == 1
