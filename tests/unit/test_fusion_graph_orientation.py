"""Graph-only numbering reuses the preferred intrinsic orientation proof."""

from dataclasses import replace

import pytest

import openclatura.fusion.numbering as numbering
from openclatura.fusion.faces import select_bounded_face_model, typed_face_model
from openclatura.fusion.layout import LayoutSearchBudgetExceeded, preferred_intrinsic_layouts
from openclatura.fusion.planner import _typed_face_model
from openclatura.graph_io import read_smiles


def _context(smiles):
    mol = read_smiles(smiles)
    faces = select_bounded_face_model(mol, mol.atoms)
    assert faces is not None
    return mol, faces


@pytest.mark.parametrize(
    "smiles",
    [
        "C1N=COC2=NON=C12",
        "c1ccc2ccccc2c1",
        "c1ccc2ncccc2c1",
        "O1C2=C(C=C1)C=CS2",
        "S1C=2N(C=C1)C=CN2",
        "c1ccc2cc3ccccc3cc2c1",
        "C1C=CC2=C1C1=CC=CC=C1C=1C=CC=CC21",
    ],
)
@pytest.mark.parametrize("reverse_faces", [False, True])
def test_graph_numbering_preserves_all_layout_preferred_maps(smiles, reverse_faces):
    mol, faces = _context(smiles)
    if reverse_faces:
        faces = replace(faces, faces=tuple(reversed(faces.faces)))
    face_model = typed_face_model(mol, faces)
    layouts = preferred_intrinsic_layouts(face_model)
    assert layouts
    explicit = numbering.completed_system_numbering_selection(mol, faces, face_model=face_model, layouts=layouts)

    graph = numbering.completed_system_numbering_selection(mol, faces)

    assert graph.accepted
    assert graph.accepted == tuple(
        replace(candidate, layout_index=None, start_face_id=None, start_atom=None) for candidate in explicit.accepted
    )
    assert graph.rejected == ()


def test_graph_numbering_abstains_without_preferred_layouts(monkeypatch):
    mol, faces = _context("c1ccc2ccccc2c1")
    monkeypatch.setattr(numbering, "preferred_intrinsic_layouts", lambda model: ())

    assert numbering.completed_system_numberings(mol, faces) == ()


def test_graph_numbering_propagates_layout_budget_exhaustion(monkeypatch):
    mol, faces = _context("c1ccc2ccccc2c1")

    def exhausted(model):
        raise LayoutSearchBudgetExceeded(1)

    monkeypatch.setattr(numbering, "preferred_intrinsic_layouts", exhausted)

    with pytest.raises(LayoutSearchBudgetExceeded, match="budget of 1"):
        numbering.completed_system_numberings(mol, faces)


def test_explicit_layout_numbering_does_not_repeat_layout_search(monkeypatch):
    mol, faces = _context("C1N=COC2=NON=C12")
    face_model = typed_face_model(mol, faces)
    layouts = preferred_intrinsic_layouts(face_model)
    expected = numbering.completed_system_numbering_selection(mol, faces, face_model=face_model, layouts=layouts)

    def unexpected(model):
        pytest.fail("caller-supplied layouts must not trigger graph-only orientation search")

    monkeypatch.setattr(numbering, "preferred_intrinsic_layouts", unexpected)

    assert (
        numbering.completed_system_numbering_selection(mol, faces, face_model=face_model, layouts=layouts) == expected
    )


def test_graph_numbering_preserves_peripheral_only_support_boundary(monkeypatch):
    mol, faces = _context("c1cc2ccc3cccc4ccc(c1)c2c34")
    assert faces.interior_atoms

    def unexpected(model):
        pytest.fail("graph-only compatibility path must reject interior atoms before layout search")

    monkeypatch.setattr(numbering, "preferred_intrinsic_layouts", unexpected)

    assert numbering.completed_system_numberings(mol, faces) == ()


def test_graph_numbering_abstains_for_a_single_ring():
    mol, faces = _context("c1ccccc1")

    assert numbering.completed_system_numberings(mol, faces) == ()


def test_planner_and_numbering_share_typed_face_model_constructor():
    assert _typed_face_model is typed_face_model
    assert numbering.typed_face_model is typed_face_model
