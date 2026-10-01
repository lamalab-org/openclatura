"""Fixed citation locants must distinguish embeddings from different skeletons."""

import random

import pytest
from rdkit import Chem

from openclatura.fusion.citation_numbering import numbered_parent_graphs_agree
from openclatura.graph_io import read_rdkit_mol
from openclatura.locants import SystemLocant


def _cycle(*, nitrogen=False, charged=False, double=False):
    graph = Chem.RWMol()
    for index in range(6):
        atom = Chem.Atom("N" if nitrogen and index == 0 else "C")
        if charged and index == 0:
            atom.SetFormalCharge(1)
        graph.AddAtom(atom)
    for index in range(6):
        order = Chem.BondType.DOUBLE if double and index == 0 else Chem.BondType.SINGLE
        graph.AddBond(index, (index + 1) % 6, order)
    result = graph.GetMol()
    Chem.SanitizeMol(result)
    return result


@pytest.mark.parametrize("seed", (7, 73, 20483))
@pytest.mark.parametrize("kind", ("automorphic", "different_edges", "heteroatom", "charge", "pi_state"))
def test_numbered_graph_agreement_is_permutation_invariant(seed, kind):
    graph = _cycle(nitrogen=kind == "heteroatom", charged=kind == "charge", double=kind == "pi_state")
    first = {atom: SystemLocant(atom + 1) for atom in range(6)}
    second = {atom: SystemLocant((atom + 1) % 6 + 1) for atom in range(6)}
    if kind == "different_edges":
        second = first.copy()
        second[0], second[1] = second[1], second[0]
    order = list(range(6))
    random.Random(seed).shuffle(order)
    mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
    maps = tuple({new: mapping[old] for new, old in enumerate(order)} for mapping in (first, second))
    expected = kind in {"automorphic", "charge", "pi_state"}
    assert numbered_parent_graphs_agree(mol, maps) is expected
    assert numbered_parent_graphs_agree(mol, reversed(maps)) is expected
    assert numbered_parent_graphs_agree(mol, maps[:1])


def test_empty_numbering_set_has_no_conflict():
    assert numbered_parent_graphs_agree(read_rdkit_mol(_cycle()), ())


def test_observed_nitrogen_charge_does_not_discriminate_parent_topology():
    graph = _cycle(nitrogen=True, charged=True)
    graph.GetAtomWithIdx(3).SetAtomicNum(7)
    Chem.SanitizeMol(graph)
    mol = read_rdkit_mol(graph)
    first = {atom: SystemLocant(atom + 1) for atom in range(6)}
    second = {atom: SystemLocant((atom + 3) % 6 + 1) for atom in range(6)}
    assert mol.atoms[0].charge == 1
    assert mol.atoms[3].charge == 0
    assert numbered_parent_graphs_agree(mol, (first, second))
