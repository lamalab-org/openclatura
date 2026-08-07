import pytest

from openclatura import name_smiles
from openclatura.chains import find_ring_systems
from openclatura.graph_io import read_smiles
from openclatura.von_baeyer import find_von_baeyer_candidates

AROMATIC_KEKULE_PAIRS = (
    (
        "c1ccc2cc3occc3cc2c1",
        "O1C2=C(C=C1)C=C1C=CC=CC1=C2",
        "tricyclo[7.4.0.0^{3,7}]",
    ),
    (
        "c1ccc2c(c1)ccc1ccsc12",
        "S1C2=C(C=C1)C=CC1=CC=CC=C12",
        "tricyclo[7.4.0.0^{2,6}]",
    ),
    (
        "c1ccc2cc3ocnc3cc2c1",
        "O1C=NC2=C1C=C1C=CC=CC1=C2",
        "tricyclo[7.4.0.0^{3,7}]",
    ),
)


def _audited_descriptors(smiles: str) -> tuple[str, ...]:
    mol = read_smiles(smiles)
    ring = find_ring_systems(mol)[0]
    atoms = set(ring.atoms)
    edges = {
        tuple(sorted((bond.u, bond.v)))
        for bond in mol.bonds.values()
        if bond.u in atoms and bond.v in atoms
    }
    return tuple(candidate.descriptor for candidate in find_von_baeyer_candidates(mol, atoms, edges))


@pytest.mark.parametrize(("aromatic", "kekule", "descriptor"), AROMATIC_KEKULE_PAIRS)
def test_audited_polycycle_descriptor_is_input_form_invariant(aromatic, kekule, descriptor):
    aromatic_descriptors = _audited_descriptors(aromatic)
    kekule_descriptors = _audited_descriptors(kekule)

    assert aromatic_descriptors[0] == descriptor
    assert kekule_descriptors[0] == descriptor


@pytest.mark.parametrize(("aromatic", "kekule", "_descriptor"), AROMATIC_KEKULE_PAIRS)
def test_public_polycycle_name_is_input_form_invariant(aromatic, kekule, _descriptor):
    assert name_smiles(aromatic) == name_smiles(kekule)
