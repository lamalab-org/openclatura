"""Charge suffixes follow the selected, audited graph numbering (index130338)."""

from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import assembly_charge, name_mol, opsin_available


@pytest.fixture
def charged_parts(monkeypatch):
    captured = []
    original = assembly_charge.fusion_parent_charge_name_operations

    def capture(parts):
        if parts.parent_hydride is not None and parts.parent_hydride.is_fusion_parent:
            captured.append(deepcopy(parts))
        return original(parts)

    monkeypatch.setattr(assembly_charge, "fusion_parent_charge_name_operations", capture)
    result = name_mol(Chem.MolFromSmiles("[NH3+][C-]1C=CC2=NON=C12"))
    assert result.ok
    return captured[0]


def test_charge_operations_follow_selected_leaf_without_mutating_audited_plan(charged_parts):
    plan = charged_parts.parent_hydride.fusion_plan
    before = plan.charge_operations
    operation = before[0]
    charge = charged_parts.parent_charges[0]
    assert operation.atom_id == charge.atom_id
    assert len(plan.numbering.input_locant_maps) == 1
    assert str(operation.locant) == charge.locant
    assert charged_parts.parent_atom_ids_by_locant[charge.locant] == operation.atom_id
    rendered = assembly_charge.fusion_parent_charge_name_operations(charged_parts)
    assert rendered[0].locants == (charge.locant,)
    assert rendered[0].suffix == "ide"
    assert plan.charge_operations == before


@pytest.mark.parametrize("corruption", ["incomplete", "unapproved", "locant", "atom", "symbol", "charge"])
def test_charge_rebinding_rejects_unproved_metadata(charged_parts, corruption):
    parts = charged_parts
    if corruption == "incomplete":
        del parts.parent_atom_ids_by_locant["1"]
    elif corruption == "unapproved":
        mapping = parts.parent_atom_ids_by_locant
        mapping["1"], mapping["2"] = mapping["2"], mapping["1"]
    else:
        changes = {
            "locant": {"locant": "6"},
            "atom": {"atom_id": -1},
            "symbol": {"symbol": "N"},
            "charge": {"charge": 1},
        }
        parts.parent_charges[0] = replace(parts.parent_charges[0], **changes[corruption])
    with pytest.raises(ValueError, match="fusion parent charge spelling"):
        assembly_charge.fusion_parent_charge_name_operations(parts)


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("[NH3+][C-]1C=CC2=NON=C12", "cyclopenta[c][1,2,5]oxadiazol-4-ide-4-aminium"),
        ("[NH2+](C)[C-]1C=CC2=NON=C12", "N-methylcyclopenta[c][1,2,5]oxadiazol-4-ide-4-aminium"),
    ],
)
def test_selected_charge_numbering_is_opsin_exact(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    orders = [list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))]
    orders.append(orders[0][3:] + orders[0][:3])
    for order in orders:
        result = name_mol(Chem.RenumberAtoms(mol, order), verify_opsin=True)
        assert result.ok
        assert result.name == expected
        assert result.opsin_check.status == "matched"
        decoded = Chem.MolFromSmiles(result.opsin_check.opsin_smiles)
        assert Chem.MolToSmiles(decoded) == Chem.MolToSmiles(mol)
        assert sorted(
            (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in decoded.GetAtoms()
        ) == sorted((atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms())
