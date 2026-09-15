"""Shared skeletal atoms and explicit HW locants have one spiro scope."""

from copy import deepcopy

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.assembly_parts import AssemblyParts, SubstituentItem
from openclatura.assembly_spiro import (
    _deduplicate_shared_replacements,
    _project_locanted_hw_component,
    spiro_assembly_from_parts,
)
from openclatura.opsin_verify import verify_with_opsin
from openclatura.spiro_assembly import SpiroAssembly

CASES = (
    pytest.param(
        "C=Cc1ccc2c(c1)[B-]1(OC(C)(C)C(C)(C)O1)[N+](C)(C)C2",
        "boranuida",
        id="pubchem-1571",
    ),
    pytest.param(
        "COc1ccc2c(c1)C1(COC(N)=N1)c1cc(-c3cnccc3F)ccc1O2",
        "1-oxa-3-azacyclopent-2-ene",
        id="pubchem-22425",
    ),
    pytest.param(
        "COc1ccc2nc3c(cc2c1)CC1(C(=O)N(C)C(=O)N(C)C1=O)[C@H]1N3[C@H]2CC(C)(C)C[C@@]1(C)C2",
        "spiro[1,3-diazinane-5,6'",
        id="pubchem-58275-control",
    ),
)


@pytest.mark.parametrize("smiles,fragment", CASES)
def test_public_names_preserve_atom_order_invariance(smiles, fragment):
    mol = Chem.MolFromSmiles(smiles)
    result = name_mol(mol, include_trace=True, verify_self=True)
    reverse = name_mol(
        Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), include_trace=True, verify_self=True
    )
    assert result.error is None
    assert reverse.error is None
    assert result.self_audit.coverage is not None
    assert not result.self_audit.coverage.unnamed_atoms
    assert reverse.self_audit.coverage is not None
    assert not reverse.self_audit.coverage.unnamed_atoms
    assert fragment in result.name
    assert reverse.name == result.name


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles,fragment", CASES)
def test_public_names_roundtrip_exactly(smiles, fragment):
    result = name_mol(Chem.MolFromSmiles(smiles), include_trace=True, verify_self=True)
    assert result.error is None
    assert result.self_audit.coverage is not None
    assert not result.self_audit.coverage.unnamed_atoms
    assert fragment in result.name
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip


def component(symbol, atoms, charge=0):
    return AssemblyParts(
        parent_length=3,
        is_ring=True,
        parent_atom_ids=set(atoms),
        parent_atom_ids_by_locant=dict(zip(("1", "2", "3"), atoms)),
        parent_atom_symbols_by_locant={"1": symbol, "2": "C", "3": "C"},
        parent_atom_charges_by_locant={"1": charge, "2": 0, "3": 0},
        parent_bond_orders_by_locants={("1", "2"): 1, ("2", "3"): 1, ("1", "3"): 1},
        a_prefixes=[SubstituentItem("bora" if symbol == "B" else "sila", ["1"], atom_ids={atoms[0]})],
    )


@pytest.mark.parametrize("symbol,charge,expected", [("B", -1, "boranuida"), ("Si", 0, "sila")])
def test_shared_replacement_is_owned_once_by_graph_identity(symbol, charge, expected):
    parent = component(symbol, (17, 18, 19), charge)
    local = component(symbol, (17, 28, 29), charge)
    original = deepcopy(local)
    side = SpiroAssembly("1", "1", "cyclopropane", side_parts=local)
    projected = _deduplicate_shared_replacements(parent, side)
    assert parent.a_prefixes == []
    assert projected.side_parts.a_prefixes[0].name == expected
    assert projected.side_parts.a_prefixes[0].atom_ids == {17}
    assert local == original
    assert _deduplicate_shared_replacements(parent, projected) == projected


def test_same_replacement_on_different_atoms_is_not_deduplicated():
    parent = component("Si", (17, 18, 19))
    local = component("Si", (27, 28, 29))
    side = SpiroAssembly("1", "1", "cyclopropane", side_parts=local)
    original = deepcopy(parent)
    assert _deduplicate_shared_replacements(parent, side) == side
    assert parent == original


def test_grouped_replacements_keep_nonjunction_atoms_and_charge_scope():
    parent = component("B", (17, 18, 19), -1)
    local = component("B", (17, 28, 29), -1)
    parent.parent_atom_symbols_by_locant["2"] = "B"
    local.parent_atom_symbols_by_locant["2"] = "B"
    parent.a_prefixes = [SubstituentItem("bora", ["1", "2"], atom_ids={17, 18})]
    local.a_prefixes = [SubstituentItem("bora", ["1", "2"], atom_ids={17, 28})]
    side = SpiroAssembly("1", "1", "cyclopropane", side_parts=local)
    projected = _deduplicate_shared_replacements(parent, side)
    assert [(item.name, item.locants, item.atom_ids) for item in parent.a_prefixes] == [("bora", ["2"], {18})]
    assert [(item.name, item.locants, item.atom_ids) for item in projected.side_parts.a_prefixes] == [
        ("bora", ["2"], {28}),
        ("boranuida", ["1"], {17}),
    ]
    assert projected.side_parts.a_prefixes[1].charge_atom_ids == {17}


def test_shared_anion_requires_combined_bond_state_for_hydride_addition():
    parent = component("B", (17, 18, 19), -1)
    local = component("B", (17, 28, 29), -1)
    local.parent_bond_orders_by_locants[("1", "2")] = 2
    side = SpiroAssembly("1", "1", "cyclopropane", side_parts=local)
    projected = _deduplicate_shared_replacements(parent, side)
    assert projected.side_parts.a_prefixes[0].name == "bora"


@pytest.mark.parametrize("symbol,retained", [("O", "1,3-oxazole"), ("S", "1,3-thiazole")])
def test_hw_projection_uses_numbered_cycle_and_preserves_side_snapshot(symbol, retained):
    local = AssemblyParts(
        parent_length=5,
        is_ring=True,
        retained_name=retained,
        parent_atom_ids={10, 11, 12, 13, 14},
        parent_atom_ids_by_locant={str(i + 1): i + 10 for i in range(5)},
        parent_atom_symbols_by_locant={"1": symbol, "2": "C", "3": "N", "4": "C", "5": "C"},
        parent_bond_orders_by_locants={
            ("1", "2"): 1,
            ("2", "3"): 2,
            ("3", "4"): 1,
            ("4", "5"): 1,
            ("1", "5"): 1,
        },
    )
    original = deepcopy(local)
    side = spiro_assembly_from_parts(local, "4")
    assert local == original
    assert side.side_parent_name == "cyclopent-2-ene"
    assert side.side_parts.parent_atom_ids_by_locant == original.parent_atom_ids_by_locant
    assert side.side_parts.unsaturations[0].atom_ids == {11, 12}
    assert [item.locants for item in side.side_parts.a_prefixes] == [["1"], ["3"]]
    broken = deepcopy(local)
    broken.parent_bond_orders_by_locants.pop(("1", "5"))
    _project_locanted_hw_component(broken)
    assert broken.retained_name == retained
