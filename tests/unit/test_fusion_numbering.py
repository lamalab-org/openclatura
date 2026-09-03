from rdkit import Chem

from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.layout import preferred_intrinsic_layouts
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.numbering import (
    completed_system_numbering_selection,
    completed_system_numberings,
    observed_parent_matches_bond_model,
    parent_bond_model,
)
from openclatura.fusion.planner import _typed_face_model, plan_fusion_parent
from openclatura.graph_io import read_smiles


def test_completed_numbering_assigns_fusion_suffixes_to_shared_carbons():
    mol = read_smiles("c1ccc2ccccc2c1")
    faces = select_bounded_face_model(mol, mol.atoms)

    assert faces is not None
    numberings = completed_system_numberings(mol, faces)

    assert numberings
    for numbering in numberings:
        values = set(numbering.string_map.values())
        assert {"4a", "8a"} <= values
        assert set(numbering.string_map) == set(mol.atoms)
        assert len(values) == len(mol.atoms)


def test_parent_bond_model_accepts_input_kekule_assignment():
    mol = read_smiles("c1ccc2ccccc2c1")

    model = parent_bond_model(mol, mol.atoms)

    assert model.maximum_non_cumulative_double_bonds == 5
    assert observed_parent_matches_bond_model(mol, model)


def test_completed_numbering_is_invariant_to_input_atom_order():
    left = read_smiles("c1ccc2ncccc2c1")
    right = read_smiles("c1cc2cccnc2cc1")
    left_faces = select_bounded_face_model(left, left.atoms)
    right_faces = select_bounded_face_model(right, right.atoms)

    assert left_faces is not None and right_faces is not None
    left_symbols = [
        sorted((str(locant), left.atoms[atom].symbol) for atom, locant in numbering.atom_to_locant)
        for numbering in completed_system_numberings(left, left_faces)
    ]
    right_symbols = [
        sorted((str(locant), right.atoms[atom].symbol) for atom, locant in numbering.atom_to_locant)
        for numbering in completed_system_numberings(right, right_faces)
    ]

    assert left_symbols == right_symbols


def _layout_numbering_selection(smiles: str):
    mol = read_smiles(smiles)
    bounded = select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None
    face_model = _typed_face_model(mol, bounded)
    layouts = preferred_intrinsic_layouts(face_model)
    selection = completed_system_numbering_selection(
        mol,
        bounded,
        face_model=face_model,
        layouts=layouts,
    )
    return mol, bounded, face_model, layouts, selection


def test_layout_derived_numbering_starts_at_uppermost_rightmost_face_and_runs_clockwise():
    _, bounded, face_model, layouts, selection = _layout_numbering_selection("O1C2=C(C=C1)C=CS2")

    assert len(selection.accepted) == 1
    selected = selection.accepted[0]
    assert selected.layout_index is not None
    layout = layouts[selected.layout_index]
    centers = {face: (x, y) for face, x, y in layout.face_positions}
    expected_face = max(centers, key=lambda face: (centers[face][1], centers[face][0]))
    positions = {atom: (x, y) for atom, x, y in layout.atom_positions}
    fusion_atoms = {atom for atom in bounded.atom_ids if sum(atom in face.atoms for face in bounded.faces) > 1}
    face = next(item for item in face_model.faces if item.id == expected_face)
    clockwise = selected.perimeter
    expected_starts = [
        atom
        for index, atom in enumerate(clockwise)
        if atom in face.atom_cycle
        and atom not in fusion_atoms
        and clockwise[index - 1] in face.atom_cycle
        and clockwise[index - 1] in fusion_atoms
    ]
    signed_area = sum(
        positions[left][0] * positions[right][1] - positions[right][0] * positions[left][1]
        for left, right in zip(selected.perimeter, selected.perimeter[1:] + selected.perimeter[:1])
    )

    assert selected.start_face_id == expected_face
    assert expected_starts == [selected.start_atom]
    assert selected.start_atom == selected.perimeter[0]
    assert signed_area < 0


def test_completed_numbering_starts_at_counterclockwise_end_of_long_nonfusion_arc():
    mol, _, _, _, selection = _layout_numbering_selection(
        "C1C=CC2=C1C1=CC=CC=C1C=1C=CC=CC21"
    )

    assert selection.accepted
    saturated_carbon = next(atom for atom, value in mol.atoms.items() if value.total_h_count == 2)
    assert {numbering.string_map[saturated_carbon] for numbering in selection.accepted} == {"1"}


def test_layout_numbering_records_locant_losing_reflections_in_the_plan_proof():
    mol = read_smiles("S1C=2N(C=C1)C=CN2")

    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionConfirmed)
    proof = result.plan.numbering
    assert proof.selected_layout.atom_positions
    assert proof.rejected_numberings
    assert all(
        "loses the ordered completed-system locant criteria" in item.reason for item in proof.rejected_numberings
    )


def test_layout_derived_numbering_is_invariant_to_input_atom_order():
    smiles = "c1cc2ccsc2o1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    renumbered = Chem.RenumberAtoms(rdkit_mol, list(reversed(range(rdkit_mol.GetNumAtoms()))))
    renumbered_smiles = Chem.MolToSmiles(renumbered, canonical=False)
    left, *_left_context, left_selection = _layout_numbering_selection(smiles)
    right, *_right_context, right_selection = _layout_numbering_selection(renumbered_smiles)

    left_maps = [
        sorted((str(locant), left.atoms[atom].symbol) for atom, locant in numbering.atom_to_locant)
        for numbering in left_selection.accepted
    ]
    right_maps = [
        sorted((str(locant), right.atoms[atom].symbol) for atom, locant in numbering.atom_to_locant)
        for numbering in right_selection.accepted
    ]

    assert left_maps == right_maps
