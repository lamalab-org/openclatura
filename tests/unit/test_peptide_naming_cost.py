"""Ring perception and rank refinement do linear work per molecule.

Each residue of a peptide names the rest of the chain as a molecule of its
own, so any per-molecule step that is quadratic in the atom count makes a
peptide of n residues cost n³ (lamalab-org/openclatura#136).  The work is
counted in graph lookups rather than timed, so the tests do not depend on the
machine.
"""

import pytest

from openclatura import name_smiles, opsin_available, verify_with_opsin
from openclatura.canonical_ranks import canonical_ranks
from openclatura.chains import _cyclic_atoms_within, get_cyclic_atoms
from openclatura.graph_io import read_smiles

ALANINE = "N[C@@H](C)C(=O)"


def _polyalanine_name(residues: int) -> str:
    prefix = "(2S)-2-aminopropanamido"
    for _ in range(residues - 2):
        prefix = f"(2S)-2-({prefix})propanamido"
    return f"(2S)-2-({prefix})propanoic acid"


def _count_calls(monkeypatch, mol, method: str) -> list[int]:
    calls = [0]
    original = getattr(mol, method)

    def counted(*args):
        calls[0] += 1
        return original(*args)

    monkeypatch.setattr(mol, method, counted)
    return calls


@pytest.mark.parametrize(
    ("smiles", "cyclic"),
    [
        ("CCCCC", set()),
        ("C1CC1CC1CCC1", {0, 1, 2, 4, 5, 6, 7}),
        ("c1ccccc1-c1ccccc1", set(range(12))),
        ("C1CCC2(CC1)CCCC2", set(range(10))),
        ("C1CC2CCC1C2", set(range(7))),
        ("C1CCCCCCCCCCC1CCO", set(range(12))),
    ],
)
def test_cyclic_atoms_are_the_atoms_on_a_ring(smiles, cyclic):
    mol = read_smiles(smiles)
    assert get_cyclic_atoms(mol) == cyclic


def test_excluding_an_atom_opens_only_its_ring():
    mol = read_smiles("C1CC1CC1CCC1")
    assert get_cyclic_atoms(mol, {5}) == {0, 1, 2}


def test_cyclic_atom_search_visits_each_atom_once(monkeypatch):
    # Far longer than the recursion limit, so the search must not recurse.
    mol = read_smiles("C" * 2000 + "1CCCCC1")
    calls = _count_calls(monkeypatch, mol, "get_neighbors")
    assert _cyclic_atoms_within(mol, set(mol.atoms)) == set(range(1999, 2005))
    assert calls[0] <= len(mol.atoms)


def test_rank_refinement_looks_up_each_bond_a_fixed_number_of_times(monkeypatch):
    # An asymmetric chain takes about one refinement round per two atoms.
    mol = read_smiles("C" * 200 + "O")
    calls = _count_calls(monkeypatch, mol, "get_bond")
    ranks = canonical_ranks(mol)
    assert len(set(ranks.values())) == len(mol.atoms)
    assert calls[0] <= 4 * len(mol.bonds)


@pytest.mark.parametrize("residues", [2, 3, 12])
def test_polyalanine_names(residues):
    assert name_smiles(ALANINE * residues + "O") == _polyalanine_name(residues)


@pytest.mark.opsin
@pytest.mark.parametrize("residues", [2, 3, 12])
def test_polyalanine_names_round_trip(residues):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    smiles = ALANINE * residues + "O"
    check = verify_with_opsin(_polyalanine_name(residues), smiles)
    assert check.status == "matched", check.to_dict()
