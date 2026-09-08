"""Seed enumeration must not collapse to an absolute-face-ID growth order."""

import random

import pytest
from rdkit import Chem

from openclatura.fusion.faces import select_bounded_face_model, typed_face_model
from openclatura.fusion.layout import preferred_intrinsic_layouts
from openclatura.fusion.numbering import completed_system_numbering_selection
from openclatura.graph_io import read_rdkit_mol

CORE = "C1=CC2=C3C4=C(C2C=C1)C1C=CC=CC1=C4c1ccccc13"


def _numbering_maps(graph, order):
    mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
    bounded = select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None
    model = typed_face_model(mol, bounded)
    layouts = preferred_intrinsic_layouts(model)
    selection = completed_system_numbering_selection(
        mol, bounded, face_model=model, layouts=layouts, defer_indicated_hydrogen=True
    )
    inverse = {original: current for current, original in enumerate(order)}
    return {
        tuple(candidate.string_map[inverse[atom]] for atom in range(graph.GetNumAtoms()))
        for candidate in selection.accepted
    }


@pytest.mark.parametrize("seed", [None, 0, 17, 53, 91, 137])
def test_tied_layout_numberings_survive_atom_relabelling(seed):
    graph = Chem.MolFromSmiles(CORE)
    original = list(range(graph.GetNumAtoms()))
    order = original.copy()
    if seed is None:
        order.reverse()
    else:
        random.Random(seed).shuffle(order)

    expected = _numbering_maps(graph, original)
    assert len(expected) == 6
    assert _numbering_maps(graph, order) == expected
    assert any((numbering[6], numbering[9]) == ("4a", "4c") for numbering in expected)
