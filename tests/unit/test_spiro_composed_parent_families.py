"""Graph-numbered spiro suffix, hydrogen and three-component composition."""

import random
from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.assembly_parts import AssemblyParts, PrincipalGroupItem, SubstituentItem
from openclatura.assembly_spiro import (
    _format_side_suffixes,
    _merge_terminal_and_side_suffixes,
    _prime_central_spiro_parts,
    spiro_assembly_from_parts,
    split_spiro_substituents,
)
from openclatura.name_operations import HydroOperation
from openclatura.spiro_assembly import SpiroAssembly

CASES = (
    (1258, "CC[C@]12CCC3C(C1CC[C@@]21C=CC(=O)O1)[C@@H](C1CC1)CC1=CC(=O)CC[C@@H]13"),
    (27384, "C=C1C[C@@]2(C(=O)N(C(=O)OC)c3c2ccc(OC)c3OC)C(C)(C)[C@@H]1C#N"),
    (33408, "COC1CCC2(CC1)Cc1ccc(-c3ccc(F)cc3F)cc1C21N=C(C)C(N)=N1"),
    (67589, "CC[C@]12CCC3C4=C(C=C(N)CC4)C4(CC4)CC3C1C1CC1[C@@]21CCC(=O)O1"),
    (70448, "CC(C)(F)COc1cnc2c(c1)[C@]1(COC(N)=N1)c1cc(-c3cccc(C#N)c3)ccc1O2"),
    (72633, "C#CC1=NC2(C(=O)C(CC)C1C)c1ccc(F)cc1CC21CCC(OC)CC1"),
    (79169, "NC1=N[C@]2(CO1)c1cc(O)ccc1Oc1ccc(-c3cncnc3)cc12"),
    (83075, "CCC(C)C1=NC2(C(=O)N1C(C)C)c1cc(CC(C)C)ccc1CC21CCCCC1"),
    (83559, "Cc1ncccc1NC(=O)[C@H]1Cn2ccnc2C2(CCN(C(=O)NC3CCCC3)CC2)O1"),
    (84640, "CCO[C@@H]1CCC2(C)C(CC[C@@]23CCC(=O)O3)C12C=CC1=CC(=O)CCC1(C)C2"),
    (86813, "C=CC1=C(C)C(=O)OC12c1cc(Cl)c(O)c(C)c1Oc1c2cc(CCCCC(=O)O)c(O)c1Cl"),
    (94434, "COc1ccc(C(=O)Cc2ccc3c(c2)C2(COC(N)=N2)C2(CC2)CO3)nc1"),
    (95135, "O=C1OC2(OC=C(c3ccc(Cl)cc3)C2=O)c2ccccc21"),
    (21841, "CCOC(=O)N1CCN2c3cc(Cl)ccc3CC3(C(=O)OC(C)(C)OC3=O)[C@@H]2C1"),
    (50170, "CC1(C)O[C@@H]2[C@@H](CO[C@]3(NC(=S)N(c4ccccc4)C3=O)[C@H]2OC(=O)c2ccccc2)O1"),
    (65115, "C[C@H](NC(=O)OC(C)(C)C)C(=O)NNC(=O)[C@@H]1C[C@]2(NC(=O)NC2=O)c2cc(F)ccc2O1"),
    (70457, "CN1C(=O)NC(=O)C12Cc1cc3nc(C4=NO[C@]5(CCN(C(=O)OC(C)(C)C)C5)C4)[nH]c3cc1C2"),
    (74654, "C[C@@]12[C@@H](OC(=O)/C=C/c3ccccc3)C[C@]3(C)O[C@]4(CC[C@]3(C)[C@H]1CCC[C@]21CO1)COC(=O)C4"),
    (67527, "COc1ccc(N2C(=O)[C@@H]3[C@@H]4CCCN4[C@]4(C(=O)Nc5ccc(F)cc54)[C@H]3C2=O)cc1"),
    (95302, "Cc1cccc(N2C(=O)[C@@H]3[C@@H]4CCCN4[C@@]4(C(=O)Nc5c(C)cc(Cl)cc54)[C@H]3C2=O)c1C"),
    (64461, "CCCCN1C(=O)[C@H]2C3CCCN3C3(C(=O)Nc4c(CC)cccc43)[C@H]2C1=O"),
    (91294, "C=CCN1C(=O)C2(c3ccccc31)C1C(=O)N(Cc3ccccc3)C(=O)C1C1CCCN12"),
    (84545, "COc1ccc2c(c1)NC(=O)[C@]21C[C@@H]2C(=O)N3CCC[C@@H]3C(=O)N2[C@@H]1C=C(C)C"),
)


def _permuted(graph, seed):
    order = list(range(graph.GetNumAtoms()))
    if seed == "reverse":
        order.reverse()
        return Chem.RenumberAtoms(graph, order)
    random.Random(seed).shuffle(order)
    graph = Chem.RenumberAtoms(graph, order)
    rebuilt = Chem.RWMol()
    for atom in graph.GetAtoms():
        rebuilt.AddAtom(Chem.Atom(atom))
    for bond in reversed(list(graph.GetBonds())):
        rebuilt.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    for atom in graph.GetAtoms():
        before = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        after = [before.index(neighbor.GetIdx()) for neighbor in rebuilt.GetAtomWithIdx(atom.GetIdx()).GetNeighbors()]
        if sum(after[i] > after[j] for i in range(len(after)) for j in range(i + 1, len(after))) % 2:
            rebuilt.GetAtomWithIdx(atom.GetIdx()).InvertChirality()
    for bond in graph.GetBonds():
        copied = rebuilt.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        copied.SetBondDir(bond.GetBondDir())
        if bond.GetStereoAtoms():
            copied.SetStereoAtoms(*bond.GetStereoAtoms())
        copied.SetStereo(bond.GetStereo())
    Chem.SanitizeMol(rebuilt)
    Chem.AssignStereochemistry(rebuilt, cleanIt=True, force=True)
    return rebuilt.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("index,smiles", CASES, ids=[str(index) for index, _ in CASES])
def test_reported_spiro_families_exact_default_and_atom_bond_permutations(index, smiles):
    if not opsin_available():
        pytest.skip("OPSIN is unavailable")
    original = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(original)
    names = []
    for graph in (original, _permuted(original, "reverse"), _permuted(original, 17)):
        assert Chem.MolToSmiles(graph) == canonical
        result = name_mol(graph)
        assert result.error is None, result
        assert "spiro[" in result.name
        assert "3H-1H-" not in result.name
        if index in {33408, 72633, 83075}:
            assert "cyclopentabenzene" in result.name
        if index == 84640:
            assert "benzobenzene" in result.name and "cyclopentabenzene" in result.name
        check = verify_with_opsin(result.name, canonical, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip == canonical
        names.append(result.name)
    assert len(set(names)) == 1


@pytest.mark.parametrize(
    "suffixes,expected",
    [
        ([("4'", "one"), ("2'", "one")], "-2',4'-dione"),
        ([("2'", "one"), ("2''", "one")], "-2',2''-dione"),
        ([("3'", "ol"), ("5'", "one"), ("2'", "ol")], "-2',3'-diol-5'-one"),
    ],
)
def test_side_suffixes_group_same_kind_without_losing_component_primes(suffixes, expected):
    assert _format_side_suffixes(suffixes) == expected


def test_side_suffix_precedes_final_substituent_suffix():
    assert _merge_terminal_and_side_suffixes("-7-yl", [("5'", "one")]) == "-5'-on-7-yl"
    assert _merge_terminal_and_side_suffixes("-2'-yl", [("2''", "one"), ("5''", "one")]) == "-2'',5''-dion-2'-yl"


def _nested_parts():
    child = AssemblyParts(
        parent_length=3,
        is_ring=True,
        parent_atom_ids={1, 3, 4},
        parent_atom_ids_by_locant={"1": 1, "2": 3, "3": 4},
    )
    side = SpiroAssembly("2", "1", "cyclopropane", side_parts=child)
    return AssemblyParts(
        parent_length=3,
        is_ring=True,
        parent_atom_ids={0, 1, 2},
        parent_atom_ids_by_locant={"1": 0, "2": 1, "3": 2},
        substituents=[SubstituentItem("", ["2"], spiro=side)],
    )


def test_nested_projection_preserves_both_numbered_component_proofs():
    parts = _nested_parts()
    original = deepcopy(parts)
    side = spiro_assembly_from_parts(parts, "1")
    assert side is not None
    assert side.continuation == parts.substituents[0].spiro
    assert side.side_substituents == ()
    assert side.side_parts == original == parts


@pytest.mark.parametrize("corruption", ["no_map", "extra_shared", "wrong_join", "same_join", "branch", "depth"])
def test_nested_projection_rejects_missing_bijection_or_nonpath_topology(corruption):
    parts = _nested_parts()
    item = parts.substituents[0]
    side = item.spiro
    if corruption == "no_map":
        side.side_parts.parent_atom_ids_by_locant.pop("3")
    elif corruption == "extra_shared":
        side.side_parts.parent_atom_ids.add(0)
        side.side_parts.parent_atom_ids_by_locant["4"] = 0
    elif corruption == "wrong_join":
        item.spiro = replace(side, side_locant="2")
    elif corruption == "same_join":
        item.spiro = replace(side, parent_locant="1")
    elif corruption == "branch":
        parts.substituents.append(deepcopy(item))
    else:
        item.spiro = replace(side, continuation=side)
    assert spiro_assembly_from_parts(parts, "1") is None


def test_overlapping_nested_paths_are_rejected_instead_of_dropping_a_component():
    side = spiro_assembly_from_parts(_nested_parts(), "1")
    parts = AssemblyParts(parent_length=4, substituents=[SubstituentItem("", ["1"], spiro=side)] * 2)
    with pytest.raises(ValueError, match="three components"):
        split_spiro_substituents(parts)


def test_central_component_projection_keeps_suffix_hydrogen_and_graph_identity():
    parts = AssemblyParts(
        parent_length=3,
        is_ring=True,
        attachment_locant="2",
        principal_group=PrincipalGroupItem("ketone", ["3"], atom_ids={2, 3}, bond_ids={9}),
        parent_atom_ids={0, 1, 2},
        parent_atom_ids_by_locant={"1": 0, "2": 1, "3": 2},
        parent_bond_orders_by_locants={("1", "2"): 1},
        parent_bond_ids_by_locants={("1", "2"): 7},
        hydro_operations=[HydroOperation("additive_hydrogen", locants=("1", "2"), atom_ids=(0, 1))],
        stereo_features=[("2", "R")],
        substituents=[SubstituentItem("methyl", ["1"], atom_ids={8})],
    )
    _prime_central_spiro_parts(parts)
    assert parts.attachment_locant == "2'"
    assert parts.parent_atom_ids_by_locant == {"1'": 0, "2'": 1, "3'": 2}
    assert parts.parent_bond_orders_by_locants == {("1'", "2'"): 1}
    assert parts.parent_bond_ids_by_locants == {("1'", "2'"): 7}
    assert parts.principal_group.locants == ["3'"]
    assert parts.principal_group.atom_ids == {2, 3}
    assert parts.hydro_operations[0].locants == ("1'", "2'")
    assert parts.hydro_operations[0].atom_ids == (0, 1)
    assert parts.stereo_features == [("2'", "R")]
