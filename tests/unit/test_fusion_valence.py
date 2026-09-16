from openclatura.fusion.valence import fusion_lambda_descriptors
from openclatura.graph_io import read_smiles
from openclatura.locants import SystemLocant


def test_lambda_descriptor_uses_complete_graph_bonding_and_proof_locant():
    mol = read_smiles("O=S1C=CC=C1")
    sulfur = next(atom_id for atom_id, atom in mol.atoms.items() if atom.symbol == "S")
    ring_atoms = set(mol.atoms) - {next(atom_id for atom_id, atom in mol.atoms.items() if atom.symbol == "O")}

    descriptors = fusion_lambda_descriptors(
        mol,
        ring_atoms,
        {atom_id: SystemLocant(index) for index, atom_id in enumerate(sorted(ring_atoms), start=1)},
    )

    assert len(descriptors) == 1
    assert descriptors[0].atom_id == sulfur
    assert descriptors[0].bonding_number == 4
    assert descriptors[0].text.endswith("lambda^4")


def test_standard_valence_and_charged_centers_do_not_enter_neutral_lambda_layer():
    standard = read_smiles("S1C=CC=C1")
    standard_locants = {atom_id: SystemLocant(index) for index, atom_id in enumerate(sorted(standard.atoms), start=1)}
    assert fusion_lambda_descriptors(standard, standard.atoms, standard_locants) == ()

    charged = read_smiles("[S+]1C=CC=C1")
    charged_locants = {atom_id: SystemLocant(index) for index, atom_id in enumerate(sorted(charged.atoms), start=1)}
    assert fusion_lambda_descriptors(charged, charged.atoms, charged_locants) == ()


def test_lambda_descriptor_requires_a_complete_parent_locant_map():
    mol = read_smiles("O=S1C=CC=C1")
    ring_atoms = {atom_id for atom_id, atom in mol.atoms.items() if atom.symbol != "O"}

    try:
        fusion_lambda_descriptors(mol, ring_atoms, {})
    except ValueError as exc:
        assert "missing completed-system locant" in str(exc)
    else:
        raise AssertionError("an incomplete proof map must not produce lambda annotations")
