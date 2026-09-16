"""Exact metric and distortion ranking for folded fused-ring layouts."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion import descriptor
from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.layout import (
    RING_SHAPE_TEMPLATES,
    _intrinsic_embedding_key,
    _layout_distortion,
    preferred_intrinsic_layouts,
)
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.numbering import completed_system_numbering_selection
from openclatura.fusion.planner import _typed_face_model, plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol

CASES = (
    ("028", "c1ccc2c(c1)ccc1c2c2c3c[nH]cc3ccc2c2ccc3ccc4ccc5cccnc5c4c3c21", {"10"}),
    ("031", "c1cc2ncc3c4c5c[nH]cc5ccc4c4c[nH]nc4c3c2cn1", {"6", "11"}),
    ("053", "c1cc2c(cn1)c1c[nH]cc1c1c2ccc2ccc3c4cc5c[nH]cc5cc4c4cc5c[nH]cc5cc4c3c21", {"2", "6", "18"}),
    ("058", "c1cc2c(cn1)c1c[nH]cc1c1c3conc3c3c4cc5cscc5cc4c4nscc4c3c21", {"2"}),
    ("071", "c1cc2c(ccc3c2c2ccc4c[nH]nc4c2c2c4c5cocc5ccc4c4c[nH]nc4c32)c2cscc12", {"2", "10"}),
    ("079", "c1ccc2c(c1)cnc1ccc3ncc4ncc5ccccc5c4c3c12", set()),
    ("isoindole", "c1nncc2cc3c[nH]cc3cc12", {"7"}),
)


@pytest.mark.parametrize("coordinate_system", ("cartesian", "eisenstein"))
@pytest.mark.parametrize("scales,expected", (((1, 1, 1), 0), ((1, 3, 3), 2), ((3, 1, 3), 2)))
def test_distortion_counts_resized_rings_invariant_to_rotation_and_common_scale(coordinate_system, scales, expected):
    shape = next(
        shape for shape in RING_SHAPE_TEMPLATES if shape.ring_size == 6 and shape.coordinate_system == coordinate_system
    )
    shapes = dict.fromkeys(range(3), shape)
    orders = {face: tuple(range(face * 10, face * 10 + shape.ring_size)) for face in shapes}
    positions = {
        atom: (x * scales[face], y * scales[face])
        for face in shapes
        for atom, (x, y) in zip(orders[face], shape.vertices, strict=True)
    }
    assert _layout_distortion(orders, shapes, positions, coordinate_system) == expected
    transformed = {
        atom: (20 - 7 * y, 30 + 7 * (x + y if coordinate_system == "eisenstein" else x))
        for atom, (x, y) in positions.items()
    }
    assert _layout_distortion(orders, shapes, transformed, coordinate_system) == expected


def test_eisenstein_embedding_identity_uses_the_exact_lattice_metric():
    shape = next(shape for shape in RING_SHAPE_TEMPLATES if shape.shape_id == "pentagon-eisenstein")
    positions = dict(enumerate(shape.vertices))
    rotated = {atom: (-y, x + y) for atom, (x, y) in positions.items()}
    orders, shapes = {0: tuple(positions)}, {0: shape}

    assert _intrinsic_embedding_key(orders, shapes, positions, coordinate_system="eisenstein") == (
        _intrinsic_embedding_key(orders, shapes, rotated, coordinate_system="eisenstein")
    )


@pytest.mark.parametrize("case_id,smiles,expected_h", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("reverse", (False, True))
def test_layout_numbering_prefers_undistorted_metric_correct_orientations(case_id, smiles, expected_h, reverse):
    graph = Chem.MolFromSmiles(smiles)
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    mol = read_rdkit_mol(graph)
    bounded = select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None
    model = _typed_face_model(mol, bounded)

    layouts = preferred_intrinsic_layouts(model)
    assert layouts
    selection = completed_system_numbering_selection(mol, bounded, face_model=model, layouts=layouts)

    assert selection.accepted
    for numbering in selection.accepted:
        assert {
            str(locant)
            for atom, locant in numbering.atom_to_locant
            if mol.atoms[atom].symbol == "N" and mol.atoms[atom].total_h_count == 1
        } == expected_h
    # These folded systems require the audited Cartesian fallback.
    if case_id not in {"028", "053", "079"}:
        assert all(layout.orientation_score[0] == 0 for layout in layouts)


@pytest.mark.opsin
@pytest.mark.parametrize("case_id,smiles,expected_h", CASES, ids=[case[0] for case in CASES])
def test_candidate_layouts_pass_full_audit_and_exact_opsin_without_promoting_default_cap(
    case_id, smiles, expected_h, monkeypatch
):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    monkeypatch.setattr(descriptor, "_SUPPORT", replace(descriptor._SUPPORT, maximum_tree_component_occurrences=16))
    graph = Chem.MolFromSmiles(smiles)
    mol = read_rdkit_mol(graph)

    planned = plan_fusion_parent(mol, mol.atoms, mode="general")

    assert isinstance(planned, FusionConfirmed), planned
    assert planned.plan.audit.confirmed
    assert set(map(str, planned.plan.indicated_hydrogens)) == expected_h
    names = []
    for reordered in (graph, Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))):
        result = name_mol(reordered, fusion_mode="general", include_trace=True)
        assert result.error is None
        assert result.parent_nomenclature == "systematic_fusion"
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        names.append(result.name)
    assert names[0] == names[1]
