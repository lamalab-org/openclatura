"""Charged replacement prefixes are selected by element, charge and bonding state."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.additive import add_replacement_prefixes
from openclatura.assembly_parts import AssemblyParts
from openclatura.assembly_prefixes import A_PREFIX_ORDER
from openclatura.molecule import Molecule


@pytest.mark.parametrize(
    "symbol,charge,valence,expected",
    (
        ("B", -1, 4, "boranuida"),
        ("Si", -1, 5, "silanuida"),
        ("Si", -1, 3, "sila"),
        ("Si", 0, 4, "sila"),
        ("B", 0, 3, "bora"),
    ),
)
@pytest.mark.parametrize("hydrogens", (0, 1))
def test_replacement_requires_the_registered_complete_valence(symbol, charge, valence, expected, hydrogens):
    mol = Molecule()
    mol.add_atom(symbol=symbol, idx=0, charge=charge, total_h_count=hydrogens)
    for index in range(1, valence - hydrogens + 1):
        mol.add_atom(symbol="C", idx=index, total_h_count=3)
        mol.add_bond(u=0, v=index, order=1)
    parts = AssemblyParts(parent_length=1)
    add_replacement_prefixes(mol, parts, [0], lambda _atom: "7")
    (item,) = parts.a_prefixes
    assert item.name == expected
    assert item.locants == ["7"]
    assert item.atom_ids == {0}
    assert item.charge_atom_ids == ({0} if expected in {"boranuida", "silanuida"} else set())


def test_charged_replacement_keeps_element_seniority():
    assert A_PREFIX_ORDER["silanuida"] == A_PREFIX_ORDER["sila"]
    assert A_PREFIX_ORDER["boranuida"] == A_PREFIX_ORDER["bora"]


@pytest.mark.opsin
@pytest.mark.parametrize("extension", (0, 1, 2))
def test_pentacoordinate_silicon_zwitterion_and_ligand_homologues(extension):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = Chem.RWMol(Chem.MolFromSmiles("C[Si-]12(C)Oc3ccccc3N=[N+]1c1ccccc1O2"))
    last = 0
    for _ in range(extension):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    Chem.SanitizeMol(graph)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert "silanuida" in result.name
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()
