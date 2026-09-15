import random

import pytest
from rdkit import Chem

from openclatura import name, name_mol, opsin_available
from openclatura.fusion import faces

CASES = [
    (8, 5, "decahydro-1H-cyclopenta[8]annulene"),
    (8, 6, "dodecahydrobenzo[8]annulene"),
    (9, 5, "dodecahydrocyclopenta[9]annulene"),
    (10, 5, "dodecahydro-1H-cyclopenta[10]annulene"),
    (10, 6, "tetradecahydrobenzo[10]annulene"),
    (11, 5, "tetradecahydrocyclopenta[11]annulene"),
]


def _graph(n, m):
    mol = Chem.RWMol()
    for _ in range(n + m - 2):
        mol.AddAtom(Chem.Atom("C"))
    cycles = (tuple(range(n)), (0, 1, *range(n, n + m - 2)))
    edges = {tuple(sorted(edge)) for cycle in cycles for edge in zip(cycle, cycle[1:] + cycle[:1])}
    for left, right in sorted(edges):
        mol.AddBond(left, right, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize(("n", "m", "expected"), CASES)
def test_larger_carbon_bicycles_use_annulene_parent(n, m, expected):
    result = name(Chem.MolToSmiles(_graph(n, m)), include_trace=True)
    assert result.name == expected
    assert result.parent_nomenclature == "systematic_fusion"
    assert result.pin_status == "confirmed"


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(("n", "m", "expected"), CASES)
@pytest.mark.parametrize(
    "variant", ["saturated", "unsaturated", "methyl", "methyl_unsaturated", "hydroxy", "hydroxy_unsaturated"]
)
def test_larger_carbon_bicycles_roundtrip_with_reordered_derivatives(n, m, expected, variant):
    mol = _graph(n, m)
    if "unsaturated" in variant:
        mol.GetBondBetweenAtoms(2, 3).SetBondType(Chem.BondType.DOUBLE)
    if "methyl" in variant:
        mol.AddBond(3, mol.AddAtom(Chem.Atom("C")), Chem.BondType.SINGLE)
    if "hydroxy" in variant:
        mol.AddBond(3, mol.AddAtom(Chem.Atom("O")), Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    original = name_mol(mol, verify_opsin=True)
    assert original.parent_nomenclature == "systematic_fusion"
    assert original.opsin_check is not None and original.opsin_check.status == "matched"
    if "hydroxy" in variant:
        assert f"[{n}]annulen-" in original.name
        assert original.name.endswith("-ol")
    else:
        assert f"[{n}]annulene" in original.name
    if variant == "saturated":
        assert original.name == expected
    reverse = list(reversed(range(mol.GetNumAtoms())))
    shuffled = list(range(mol.GetNumAtoms()))
    random.Random(20260908).shuffle(shuffled)
    for order in (reverse, shuffled):
        reordered = name_mol(Chem.RenumberAtoms(mol, order), verify_opsin=True)
        assert reordered.name == original.name
        assert reordered.opsin_check is not None and reordered.opsin_check.status == "matched"


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(("n", "m"), [(n, m) for n in range(7, 21) for m in range(5, n + 1)])
def test_entire_supported_two_ring_interval_roundtrips(n, m):
    result = name_mol(_graph(n, m), verify_opsin=True)
    if n == 7:
        # Complete retained parents retain precedence over component fusion.
        assert (
            result.name
            == {
                5: "decahydroazulene",
                6: "decahydro-1H-benzo[7]annulene",
                7: "dodecahydroheptalene",
            }[m]
        )
    else:
        assert result.parent_nomenclature == "systematic_fusion"
    assert result.opsin_check is not None and result.opsin_check.status == "matched"


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(("n", "m"), [(n, m) for n in (12, 15, 20) for m in (5, 6, n)])
@pytest.mark.parametrize("derivative", [False, True])
def test_larger_production_graphs_reuse_the_same_bounded_route(monkeypatch, n, m, derivative):
    def forbidden(*args, **kwargs):
        pytest.fail("larger ordinary bicycles must still bypass cycle enumeration")

    monkeypatch.setattr(faces, "enumerate_chordless_cycles", forbidden)
    mol = _graph(n, m)
    if derivative:
        mol.GetBondBetweenAtoms(2, 3).SetBondType(Chem.BondType.DOUBLE)
        mol.AddBond(3, mol.AddAtom(Chem.Atom("C")), Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    result = name_mol(mol, verify_opsin=True)
    assert result.parent_nomenclature == "systematic_fusion"
    assert f"[{n}]annulene" in result.name
    assert result.opsin_check is not None and result.opsin_check.status == "matched"
    reordered = name_mol(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), verify_opsin=True)
    assert reordered.name == result.name
    assert reordered.opsin_check is not None and reordered.opsin_check.status == "matched"


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CC1CCCCCC2CCCC2C1", "5-methyldodecahydrocyclopenta[9]annulene"),
        ("OC1CCCCCC2CCCC2C1", "dodecahydrocyclopenta[9]annulen-5-ol"),
        ("CC1CCCCCCC2CCCC2C1", "5-methyldodecahydro-1H-cyclopenta[10]annulene"),
        ("OC1CCCCCCC2CCCC2C1", "dodecahydro-1H-cyclopenta[10]annulen-5-ol"),
        (
            "CC1=CC2CCCC2CCCCCC1",
            "10-methyl-2,3,3a,4,5,6,7,8,9,11a-decahydro-1H-cyclopenta[10]annulene",
        ),
        (
            "OC1=CC2CCCC2CCCCCC1",
            "2,3,3a,6,7,8,9,10,11,11a-decahydro-1H-cyclopenta[10]annulen-5-ol",
        ),
    ],
)
def test_proved_hydrogen_numbering_preserves_low_derivative_locants(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(mol, order), verify_opsin=True)
        assert result.name == expected
        assert result.parent_nomenclature == "systematic_fusion"
        assert result.pin_status == "confirmed"
        assert result.opsin_check is not None and result.opsin_check.status == "matched"
