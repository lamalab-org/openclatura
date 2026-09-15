"""Exact, nonstandardized OPSIN evidence for generic fusion hydrogenation."""

import random

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol

CASES = [
    (
        "O=C(COc1ccc2nc3n(c(=O)c2c1)CCC3)Nc1ccc(F)cc1",
        "N-(4-fluorophenyl)-2-((9-oxo-2,3-dihydro-1H-pyrrolo[1,2-a]benzo[d]pyrimidin-7-yl)oxy)acetamide",
    ),
    (
        "O=C(NCc1ccccc1)Nc1ccc(C2=CSC3=NCCN23)cc1",
        "N-benzyl-N'-(4-(5,6-dihydroimidazo[2,1-b][1,3]thiazol-3-yl)phenyl)urea",
    ),
    ("O=c1c2ccccc2nc2n1CCC2", "2,3-dihydropyrrolo[1,2-a]benzo[d]pyrimidin-9(1H)-one"),
    ("C1=CSC2=NCCN12", "5,6-dihydroimidazo[2,1-b][1,3]thiazole"),
    ("O=c1c2ccccc2nc2n1CC(C)C2", "2-methyl-2,3-dihydropyrrolo[1,2-a]benzo[d]pyrimidin-9(1H)-one"),
    ("CC1CN2C(=NC1)SC=C2", "6-methyl-6,7-dihydro-5H-thiazolo[3,2-a]pyrimidine"),
    ("O=c1c2ccccc2nc2n1CC=C2", "pyrrolo[1,2-a]benzo[d]pyrimidin-9(1H)-one"),
]


@pytest.mark.opsin
@pytest.mark.parametrize("smiles,expected", CASES)
@pytest.mark.parametrize("mode", [FusionMode.GENERAL, FusionMode.AUDITED_PIN])
@pytest.mark.parametrize("ordering", ["original", "reversed", "shuffled"])
def test_generic_fusion_hydrogenation_exact_roundtrip(smiles, expected, mode, ordering):
    mol = Chem.MolFromSmiles(smiles)
    order = list(range(mol.GetNumAtoms()))
    if ordering == "reversed":
        order.reverse()
    elif ordering == "shuffled":
        random.Random(17).shuffle(order)
    result = name_mol(Chem.RenumberAtoms(mol, order), fusion_mode=mode, verify_opsin=True)
    assert result.name == expected
    assert result.opsin_check is not None and result.opsin_check.status == "matched"
    decoded = Chem.MolFromSmiles(result.opsin_check.opsin_smiles)
    assert decoded is not None
    # Do not accept tautomer, resonance, or standardization equivalence here.
    assert Chem.MolToSmiles(decoded, isomericSmiles=True) == Chem.MolToSmiles(mol, isomericSmiles=True)


@pytest.mark.opsin
@pytest.mark.parametrize("smiles,expected", CASES)
def test_explicit_kekule_smiles_preserves_fusion_hydrogenation(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    kekule = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)
    assert ":" not in kekule
    result = name_mol(Chem.MolFromSmiles(kekule), verify_opsin=True)
    assert result.name == expected
    assert result.opsin_check is not None and result.opsin_check.status == "matched"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result.opsin_check.opsin_smiles)) == Chem.MolToSmiles(mol)


def test_hydro_metadata_survives_recursive_carbon_branch_assembly():
    smiles, expected = CASES[1]
    mol = Chem.MolFromSmiles(smiles)
    result = name_mol(mol, include_trace=True)
    assert result.name == expected
    decisions = [
        decision
        for segment in result.trace_segments
        for decision in segment.get("nested_decisions", [])
        if decision["decision"] == "selected audited systematic fusion parent"
    ]
    assert decisions
    for decision in decisions:
        data = decision["data"]
        operations = data["derivative_operations"]["hydro"]
        assert len(operations) == 1
        operation = operations[0]
        assert operation["locants"] == ["5", "6"]
        assert [data["atom_to_locant"][atom] for atom in operation["atom_ids"]] == ["5", "6"]
        assert all(mol.GetAtomWithIdx(atom).GetSymbol() == "C" for atom in operation["atom_ids"])
        assert mol.GetBondBetweenAtoms(*operation["atom_ids"]).GetBondType() == Chem.BondType.SINGLE


@pytest.mark.parametrize("case,locants", [(0, ["2", "3"]), (1, ["5", "6"])])
def test_full_branch_tree_retains_graph_bound_hydrogenation(case, locants):
    smiles, expected = CASES[case]
    mol = Chem.MolFromSmiles(smiles)
    result = name_mol(mol, include_trace=True)
    assert result.name == expected

    def nodes(tree):
        for node in tree:
            yield node
            yield from nodes(node.get("substituents", []))

    operations = [
        operation for node in nodes(result.substituent_tree) for operation in node.get("hydro_operations", [])
    ]
    hydro = [operation for operation in operations if operation["operation_kind"] == "additive_hydrogen"]
    added = [operation for operation in operations if operation["key"] == "added_hydrogen"]
    assert len(hydro) == 1
    assert len(operations) == 1 + (case == 0)
    assert len(added) == (case == 0)
    operation = hydro[0]
    assert operation["locants"] == locants
    assert operation["operation_kind"] == "additive_hydrogen"
    assert len(operation["atom_ids"]) == 2
    assert len(operation["bond_ids"]) == 1
    assert all(mol.GetAtomWithIdx(atom).GetSymbol() == "C" for atom in operation["atom_ids"])
    bond = mol.GetBondBetweenAtoms(*operation["atom_ids"])
    assert bond.GetBondType() == Chem.BondType.SINGLE
    # Internal graph bond IDs are one-based, unlike RDKit's bond indices.
    assert operation["bond_ids"] == [bond.GetIdx() + 1]
    if added:
        assert added[0]["locants"] == ["1"]
        assert added[0]["operation_kind"] == "indicated_hydrogen"
        (atom,) = added[0]["atom_ids"]
        assert atom not in operation["atom_ids"]
        assert mol.GetAtomWithIdx(atom).GetSymbol() == "C"
        assert all(
            atom in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond_id in added[0]["bond_ids"]
            for bond in (mol.GetBondWithIdx(bond_id - 1),)
        )
