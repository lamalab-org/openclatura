from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.numbering import (
    completed_system_numberings,
    observed_parent_matches_bond_model,
    parent_bond_model,
)
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
