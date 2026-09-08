"""Spiro parent-pipeline regressions."""

from copy import deepcopy
from dataclasses import replace

import pytest

from openclatura.assembly_parts import AssemblyParts, PrincipalGroupItem, SubstituentItem
from openclatura.assembly_spiro import (
    _normalize_spiro_assembly,
    _prime_side_suffixes,
    spiro_assembly_from_parts,
    split_spiro_substituents,
)
from openclatura.fusion.model import FusionMode
from openclatura.graph_io import read_smiles
from openclatura.name_operations import HydroOperation
from openclatura.spiro_assembly import SpiroAssembly
from openclatura.spiro_subgraph import plan_graph_spiro_side, plan_substituted_fusion_spiro_side


def test_typed_side_hoisting_does_not_parse_nested_prefix_names(monkeypatch):
    import openclatura.assembly_spiro as assembly

    def reject_parse(*args, **kwargs):
        pytest.fail("typed side prefixes must not be recovered from rendered text")

    monkeypatch.setattr(assembly, "_split_side_prefix_run", reject_parse)
    branch = SubstituentItem(
        name="(2,6-difluoro-3-methylphenyl)",
        locants=["3a"],
        atom_ids={8, 9},
        bond_ids={11},
        trace_segments=[{"atoms": [8, 9]}],
        substituent_tree={"name": "phenyl", "locants": ["2", "6"]},
        outer_parentheses_optional=True,
    )
    local = AssemblyParts(parent_length=6, is_ring=True, substituents=[branch])
    original = deepcopy(local)
    side = spiro_assembly_from_parts(local, "1")
    assert local == original
    assert side.side_parts == original
    assert side.side_substituents[0].locants == ["3a'"]
    assert _normalize_spiro_assembly(side) == side
    parts = AssemblyParts(parent_length=3, substituents=[SubstituentItem("", ["1"], spiro=side)])
    normalized = split_spiro_substituents(parts)[0]
    assert normalized.side_prefixes == ()
    assert parts.substituents == [replace(branch, locants=["3a'"])]
    assert parts.substituents[0].substituent_tree == branch.substituent_tree
    assert split_spiro_substituents(parts) == []
    assert len(parts.substituents) == 1


def test_typed_replacements_are_projected_from_parts_without_parsing(monkeypatch):
    import openclatura.assembly_spiro as assembly

    def reject_parse(*args, **kwargs):
        pytest.fail("typed replacements must not be recovered from rendered text")

    monkeypatch.setattr(assembly, "_split_side_prefix_run", reject_parse)
    local = AssemblyParts(
        parent_length=5,
        is_ring=True,
        a_prefixes=[SubstituentItem("aza", ["1", "3"])],
        substituents=[SubstituentItem("methyl", ["2"], atom_ids={8})],
    )
    original = deepcopy(local)
    side = spiro_assembly_from_parts(local, "4")
    assert side.side_parent_name == "cyclopentane"
    assert side.side_prefixes == ("2'-methyl", "1',3'-diaza")
    parts = AssemblyParts(parent_length=3, substituents=[SubstituentItem("", ["1"], spiro=side)])
    normalized = split_spiro_substituents(parts)[0]
    assert normalized.side_prefixes == ("1',3'-diaza",)
    assert parts.substituents == [replace(original.substituents[0], locants=["2'"])]
    assert local == original


def test_typed_suffixes_and_stereo_are_projected_once():
    local = AssemblyParts(
        parent_length=6,
        is_ring=True,
        principal_group=PrincipalGroupItem("ketone", ["2"], atom_ids={1, 6}),
        stereo_features=[("1", "R"), ("3", "S")],
        omit_redundant_locants=False,
    )
    side = spiro_assembly_from_parts(local, "1")
    assert side.side_parent_name == "cyclohexane"
    assert side.side_suffixes == (("2", "one"),)
    assert side.side_stereo == (("3'", "S"),)
    assert _prime_side_suffixes(side.side_suffixes, "'") == [("2'", "one")]
    assert _prime_side_suffixes((("2'", "one"),), "'") == [("2'", "one")]
    assert _normalize_spiro_assembly(_normalize_spiro_assembly(side)) == side


def test_added_hydrogen_without_suffix_stays_component_local():
    operation = HydroOperation("added_hydrogen", locants=("1",), atom_ids=(0,))
    local = AssemblyParts(
        parent_length=6,
        is_ring=True,
        hydro_operations=[operation],
        omit_redundant_locants=False,
    )
    side = spiro_assembly_from_parts(local, "3")
    assert side is not None
    assert side.side_parent_name == "1H-cyclohexane"
    assert side.side_suffixes == ()
    assert side.side_parts.hydro_operations == [operation]
    assert local.hydro_operations == [operation]


@pytest.mark.parametrize("key", ["ketone", "alcohol", "amine"])
def test_compound_suffix_is_not_concatenated_into_spiro_name(key):
    local = AssemblyParts(
        parent_length=6,
        is_ring=True,
        principal_group=PrincipalGroupItem(key, ["2"]),
        hydro_operations=[HydroOperation("added_hydrogen", locants=("1",))],
    )
    original = deepcopy(local)
    assert spiro_assembly_from_parts(local, "3") is None
    assert local == original


def test_nested_spiro_wrapper_is_not_silently_hoisted_as_empty_prefix():
    nested = SpiroAssembly(parent_locant="1", side_locant="1", side_parent_name="cyclopropane")
    local = AssemblyParts(parent_length=6, substituents=[SubstituentItem("", ["1"], spiro=nested)])
    assert spiro_assembly_from_parts(local, "3") is None


def test_charged_parent_requires_shared_post_assembly_renderer():
    local = AssemblyParts(parent_length=6, is_ring=True, parent_atom_charges_by_locant={"1": -1})
    assert spiro_assembly_from_parts(local, "3") is None
    calls = []

    def renderer(parts):
        calls.append(parts)
        assert parts.parent_atom_charges_by_locant == {"1": -1}
        assert parts.substituents == []
        return "cyclohexan-1-ide"

    side = spiro_assembly_from_parts(local, "3", render_parent=renderer)
    assert side is not None
    assert side.side_parent_name == "cyclohexan-1-ide"
    assert side.side_parts.parent_atom_charges_by_locant == {"1": -1}
    assert len(calls) == 1


def test_callback_returns_normal_assembly_and_captures_unmodified_parts(monkeypatch):
    import openclatura.component_namer as component_namer
    import openclatura.namer as namer
    from openclatura.molecule import Molecule

    mol = read_smiles("C1CCCCC1")
    atoms = set(mol.atoms)
    local = AssemblyParts(
        parent_length=6,
        is_ring=True,
        parent_atom_ids=atoms,
        parent_atom_ids_by_locant={str(index + 1): atom for index, atom in enumerate(sorted(atoms))},
    )
    original = Molecule.subgraph
    subgraph_calls = []

    def subgraph(self, *args, **kwargs):
        subgraph_calls.append(kwargs)
        return original(self, *args, **kwargs)

    def assemble(_mol, parts, _path, _get_loc, **_kwargs):
        parts.front_modifiers.append("mutation after capture")
        return "cyclohexane"

    def component(side_mol, side_atoms, *, assemble_parent_name, **kwargs):
        assert side_atoms == atoms
        assert kwargs["return_tree"]
        assert assemble_parent_name(side_mol, local, sorted(atoms), lambda atom: str(atom + 1)) == "cyclohexane"
        return "cyclohexane", {}

    monkeypatch.setattr(Molecule, "subgraph", subgraph)
    monkeypatch.setattr(namer, "_assemble_parent_name", assemble)
    monkeypatch.setattr(component_namer, "name_component", component)
    side = plan_graph_spiro_side(mol, atoms, 0)
    assert side is not None
    assert side.side_parts.front_modifiers == []
    assert len(subgraph_calls) == 1
    assert not subgraph_calls[0].get("symbols")


def test_fused_side_uses_selected_nonfirst_numbering(monkeypatch):
    import openclatura.parent_pipeline as pipeline

    original = pipeline.choose_parent_numbering
    selected_maps = []
    first_maps = []

    def choose_last(mol, paths, principal, branches, maps, *args, **kwargs):
        if maps and len(maps) > 1:
            first_maps.append(maps[0])
            selected_maps.append(maps[-1])
            maps = [maps[-1]]
        return original(mol, paths, principal, branches, maps, *args, **kwargs)

    monkeypatch.setattr(pipeline, "choose_parent_numbering", choose_last)
    mol = read_smiles("C1Cc2nccnc2[C@H]1N")
    side = plan_substituted_fusion_spiro_side(mol, set(mol.atoms), 0, mode=FusionMode.AUDITED_PIN)
    assert side is not None
    assert len(selected_maps) == len(first_maps) == 1
    assert selected_maps[0] != first_maps[0]
    mapping = {atom: locant for locant, atom in side.side_parts.parent_atom_ids_by_locant.items()}
    assert mapping == selected_maps[-1]
    assert side.side_locant == mapping[0]
    parent = side.side_parts.parent_hydride
    assert parent.is_systematic_fusion
    assert list(parent.fusion_plan.numbering.string_input_locant_maps()) == [mapping]
    assert not parent.fusion_plan.numbering_variants
    assert side.side_prefixes == ("5'-amino",)
    assert side.side_stereo == (("5'", "S"),)


@pytest.mark.parametrize("smiles", ["C1Cc2ncccc2C1=O", "C1Cc2ncccc2[C@H]1N"])
def test_fused_side_uses_shared_parts_and_complete_selected_variant(smiles):
    mol = read_smiles(smiles)
    side = plan_substituted_fusion_spiro_side(mol, set(mol.atoms), 0, mode=FusionMode.AUDITED_PIN)
    assert side is not None
    parts = side.side_parts
    assert parts is not None
    mapping = {atom: locant for locant, atom in parts.parent_atom_ids_by_locant.items()}
    assert mapping[0] == side.side_locant
    assert set(mapping) == parts.parent_atom_ids
    parent = parts.parent_hydride
    assert parent is not None
    if parent.is_systematic_fusion:
        selected = parent.fusion_plan
        assert not selected.numbering_variants
        assert list(selected.numbering.string_input_locant_maps()) == [mapping]
        assert parts.parent_bond_delta == selected.derivative_state.bond_delta
    assert side.side_substituents
    assert all(item.atom_ids and item.bond_ids for item in side.side_substituents)


def test_nonfusion_bicycle_uses_graph_junction_without_symbol_marker(monkeypatch):
    from openclatura.molecule import Molecule

    original = Molecule.subgraph

    def checked_subgraph(self, *args, **kwargs):
        assert not kwargs.get("symbols")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Molecule, "subgraph", checked_subgraph)
    mol = read_smiles("C1CC2CCC1C2")
    side = plan_graph_spiro_side(mol, set(mol.atoms), 0)
    assert side is not None
    assert side.side_parts.parent_atom_ids_by_locant[side.side_locant] == 0


def test_shortcut_without_numbered_parts_does_not_invent_junction(monkeypatch):
    import openclatura.component_namer as component_namer

    monkeypatch.setattr(component_namer, "name_component", lambda *args, **kwargs: "silacyclohexane")
    mol = read_smiles("C1CCCCC1")
    assert plan_graph_spiro_side(mol, set(mol.atoms), 0) is None


def test_junction_intent_numbers_once_before_suffix_feature_assembly(monkeypatch):
    import openclatura.component_namer as component
    import openclatura.parent_pipeline as pipeline

    original = pipeline.choose_parent_numbering
    original_features = component.add_component_principal_group
    numbering = []
    features = []

    def choose(mol, paths, principal, *args, **kwargs):
        numbering.append(tuple(principal))
        return original(mol, paths, principal, *args, **kwargs)

    def add_group(mol, parts, groups, key, principal, path, get_loc):
        features.append(dict(parts.parent_atom_ids_by_locant))
        return original_features(mol, parts, groups, key, principal, path, get_loc)

    monkeypatch.setattr(pipeline, "choose_parent_numbering", choose)
    monkeypatch.setattr(component, "add_component_principal_group", add_group)
    mol = read_smiles("C1CC1O")
    side = plan_graph_spiro_side(mol, set(mol.atoms), 0)
    assert side is not None
    assert numbering == [(0,)]
    assert features == [side.side_parts.parent_atom_ids_by_locant]
    assert side.side_locant == "1"
    assert side.side_suffixes == (("2", "ol"),)
    assert side.side_parts.parent_atom_ids_by_locant["2"] == 2
