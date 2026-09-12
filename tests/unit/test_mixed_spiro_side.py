"""Substituted fused sides of mixed spiro branches (failure index1244)."""

import pytest
from rdkit import Chem

from openclatura import name, name_mol, opsin_available, verify_with_opsin
from openclatura.assembler import assemble_name_raw, assemble_name_result
from openclatura.assembly_parts import AssemblyParts, SubstituentItem
from openclatura.assembly_spiro import split_spiro_substituents
from openclatura.fusion.model import FusionMode
from openclatura.graph_io import read_rdkit_mol, read_smiles
from openclatura.spiro_subgraph import plan_substituted_fusion_spiro_side

SMILES = "Cc1nc(N2CCC3(CC2)Cc2ncccc2[C@H]3N)c(CO)nc1Sc1ccnc(N)c1Cl"
EXPECTED = (
    "(6-((2-amino-3-chloropyridin-4-yl)sulfanyl)-3-((5S)-5-aminospiro[6,7-dihydro-"
    "5H-cyclopenta[b]pyridine-6,4'-piperidine]-1'-yl)-5-methylpyrazin-2-yl)methanol"
)


def test_index1244_uses_substituted_fusion_side():
    assert name(SMILES).name == EXPECTED


@pytest.mark.parametrize("seed", range(4))
def test_index1244_atom_order_independent(seed):
    import random

    mol = Chem.MolFromSmiles(SMILES)
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    assert name_mol(Chem.RenumberAtoms(mol, order)).name == EXPECTED


@pytest.mark.parametrize(
    "smiles",
    [
        SMILES,
        SMILES.replace("[C@H]3N", "[C@@H]3N"),
        SMILES.replace("[C@H]3N", "C3N"),
        SMILES.replace("[C@H]3N", "[C@H]3NC"),
        SMILES.replace("[C@H]3N", "[C@H]3c2ccccc2"),
        SMILES.replace("[C@H]3N", "C3"),
        SMILES.replace("Cc2ncccc2", "Cc2ncc(C)cc2"),
    ],
)
def test_mixed_spiro_side_exact_opsin_roundtrip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    generated = name(smiles).name
    assert "cyclopenta[b]pyridine" in generated
    check = verify_with_opsin(generated, smiles, standardize_smiles=False)
    assert check.ok, check.to_dict()
    assert check.canonical_roundtrip == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_side_adapter_preserves_hydrogenation_branches_and_stereo():
    mol = read_smiles("C1Cc2ncccc2[C@H]1N")
    side = plan_substituted_fusion_spiro_side(mol, set(mol.atoms), 0, mode=FusionMode.AUDITED_PIN)
    assert side is not None
    assert side.side_locant == "6"
    assert side.side_parent_name == "6,7-dihydro-5H-cyclopenta[b]pyridine"
    assert side.side_prefixes == ("5'-amino",)
    assert side.side_stereo == (("5'", "S"),)


def test_substituted_fusion_side_does_not_use_silicon_marker(monkeypatch):
    import openclatura.namer as namer

    def reject_marker(*args, **kwargs):
        pytest.fail("eligible fused side reached silicon-marker component naming")

    monkeypatch.setattr(namer, "name_component", reject_marker)
    mol = read_smiles("C1Cc2ncccc2[C@H]1N")
    side = namer._spiro_subgraph_assembly(mol, 0, set(mol.atoms))
    assert side.side_locant == "6"
    assert side.side_prefixes == ("5'-amino",)


def test_side_prefix_hoisting_preserves_graph_metadata_and_bindings():
    from dataclasses import replace

    mol = read_smiles("C1Cc2ncccc2[C@H]1N")
    side = plan_substituted_fusion_spiro_side(mol, set(mol.atoms), 0, mode=FusionMode.AUDITED_PIN)
    side = replace(side, parent_locant="4")
    amino = side.side_substituents[0]
    parts = AssemblyParts(
        parent_length=6,
        is_ring=True,
        retained_name="piperidine",
        substituents=[SubstituentItem(name="", locants=["4"], spiro=side)],
    )
    rendered = assemble_name_result(parts)
    hoisted = parts.substituents[0]
    assert hoisted.atom_ids == amino.atom_ids
    assert hoisted.bond_ids == amino.bond_ids
    assert hoisted.substituent_tree == amino.substituent_tree
    assert hoisted.trace_segments == amino.trace_segments
    binding = next(binding for binding in rendered.bindings if binding.term == "amino")
    assert binding.atom_ids == amino.atom_ids
    assert binding.bond_ids == amino.bond_ids
    # P-24.5.1 cites this component first, so its locant is unprimed.
    assert binding.locants == ("5",)
    assert binding.emitted_tokens

    parts = AssemblyParts(parent_length=6, substituents=[SubstituentItem(name="", locants=["4"], spiro=side)])
    normalized = split_spiro_substituents(parts)[0]
    # P-24.5.1 cites this component first, so its locants lose the prime. Every
    # other thing the side carries has to survive that untouched.
    assert [
        replace(item, locants=[str(locant).rstrip("'") for locant in item.locants])
        for item in side.side_substituents
    ] == list(normalized.side_substituents)


def test_public_trace_keeps_side_amino_ownership():
    result = name(SMILES, include_trace=True, token_debug=True)
    assert result.name == EXPECTED
    amino = next(segment for segment in result.trace_segments if segment["key"] == "substituent:amino")
    assert amino["atoms"] == [18]
    assert amino["bonds"] == [18]
    assert result.substituent_tree
    assert result.decisions


@pytest.mark.parametrize("mode", [FusionMode.DISABLED, FusionMode.LEGACY])
def test_side_adapter_respects_fusion_policy(mode):
    mol = read_smiles("C1Cc2ncccc2[C@H]1N")
    assert plan_substituted_fusion_spiro_side(mol, set(mol.atoms), 0, mode=mode) is None


def test_fusion_side_adapter_declines_a_single_ring():
    mol = read_smiles("C1CC(N)CCC1")
    assert plan_substituted_fusion_spiro_side(mol, set(mol.atoms), 0, mode=FusionMode.AUDITED_PIN) is None


def test_fusion_side_adapter_preserves_oxo_operation_and_attachment():
    mol = read_smiles("C1Cc2ncccc2C1=O")
    side = plan_substituted_fusion_spiro_side(mol, set(mol.atoms), 0, mode=FusionMode.AUDITED_PIN)
    assert side is not None
    parts = side.side_parts
    assert parts.parent_atom_ids_by_locant[side.side_locant] == 0
    plan = parts.parent_hydride.fusion_plan
    assert plan is not None
    assert len(plan.derivative_state.oxo_operations) == 1
    operation = plan.derivative_state.oxo_operations[0]
    assert parts.parent_atom_ids_by_locant[str(operation.locant)] == operation.parent_atom_id
    assert mol.atoms[operation.oxygen_atom_id].symbol == "O"
    assert mol.bonds[operation.bond_id].order == 2
    assert any(operation.oxygen_atom_id in item.atom_ids for item in side.side_substituents)


@pytest.mark.parametrize("substitution", [None, (1, "C"), (4, "F")])
def test_graph_built_fused_ketone_sides_keep_exact_hydrogenation(substitution):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = Chem.RWMol(Chem.MolFromSmiles("C1Cc2ncccc2C1=O"))
    if substitution is not None:
        parent, symbol = substitution
        added = graph.AddAtom(Chem.Atom(symbol))
        graph.AddBond(parent, added, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    original = Chem.MolToSmiles(graph)
    orders = [list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))]
    names = []
    for order in orders:
        mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
        side = plan_substituted_fusion_spiro_side(mol, set(mol.atoms), order.index(0), mode=FusionMode.AUDITED_PIN)
        assert side is not None
        assert side.side_parts.parent_hydride.is_systematic_fusion
        assert side.side_parts.parent_atom_ids_by_locant[side.side_locant] == order.index(0)
        generated = assemble_name_raw(side.side_parts)
        check = verify_with_opsin(generated, original, standardize_smiles=False)
        assert check.ok, check.to_dict()
        assert check.canonical_roundtrip == original
        names.append(generated)
    assert names[0] == names[1]
