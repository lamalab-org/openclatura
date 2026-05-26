import pytest

from structure_to_iupac import NamingEngine, NamingIntent, OperationClass, RULES, TracePhase, analyze_smiles, name_smiles
from structure_to_iupac.assembler import AssemblyParts, ParentChargeItem, SubstituentItem, UnsaturationItem, assemble_name, post_process_name
from structure_to_iupac.assembly_charge import parent_charge_operations
from structure_to_iupac.assembly_spiro import _spiro_side_locant, extract_spiro_side_prefixes, format_spiro_core, split_spiro_substituents
from structure_to_iupac.chains import find_all_carbon_paths, find_ring_systems
from structure_to_iupac.functional_groups import PERCEPTION_DETECTORS, PERCEPTION_SPECS, PerceptionDetectorSpec, register_group_detector, register_perception_spec
from structure_to_iupac.formatting import ensure_stereo_descriptor_boundary
from structure_to_iupac.ionic_naming import (
    apply_ring_parent_nitrogen_zwitterion_stack,
    apply_parent_charge_names,
    apply_retained_parent_ide,
    apply_terminal_parent_ide,
    contains_invalid_locant_ide,
    parent_charge_sites,
)
from structure_to_iupac.locants import as_display_locant, coerce_display_numbering, locant_text, parse_locant
from structure_to_iupac.name_postprocessing import apply_connection_boundary_postprocessing, postprocessing_rule_inventory
from structure_to_iupac.namer import _spiro_subgraph_assembly, name_component, read_smiles
from structure_to_iupac.numbering import NUMBERING_CRITERIA, NumberingPreference, polycycle_numbering_key
from structure_to_iupac.parent_selection import (
    PARENT_SELECTION_CRITERIA,
    ParentCandidate,
    ParentSelection,
    ParentSeniorityProfile,
    select_principal_parent,
)
from structure_to_iupac.parent_pipeline import build_parent_assembly_plan
from structure_to_iupac.perception import PerceivedGroup, perceive_groups
from structure_to_iupac.polycycle_topology import (
    bicyclo_proof,
    build_ring_numbering,
    linear_dispiro_proof,
    monospiro_proof,
    ring_system_topology,
)
from structure_to_iupac.principal_suffixes import render_principal_suffix
from structure_to_iupac.retained_specs import retained_parent_spec
from structure_to_iupac.rules.retained import get_retained_ring
from structure_to_iupac.ring_systems import ring_system_fragment
from structure_to_iupac.ring_renderer import render_ring_descriptor, render_von_baeyer_descriptor
from structure_to_iupac.spiro_assembly import SpiroAssembly
from structure_to_iupac.special_cases import _alkyl_ligand_name, structural_replacement_parent_name
from structure_to_iupac.heteroatom_substituent_specs import ligand_prefix, unsubstituted_prefix
from structure_to_iupac.name_operations import HydroOperation, ParentSuffixOperation
from structure_to_iupac.name_bindings import postprocess_name_atom_bindings
from structure_to_iupac.naming_audit import UnnamedAtomError, assert_component_fully_named
from structure_to_iupac.naming_data import namer_rules
from structure_to_iupac.rule_layout import rule_group_specs, rule_groups, section_group_map, unassigned_sections
from structure_to_iupac.suffix_stack import SuffixStack
from structure_to_iupac.molecule import Molecule
from structure_to_iupac.assembly_parts import NameAtomBinding
from structure_to_iupac.stereo_audit import audit_stereochemistry
from structure_to_iupac.additive import add_indicated_hydrogens


def test_name_smiles_stays_plain_fast_api():
    assert name_smiles("CCO") == "ethanol"


def test_methane_is_named_without_parent_selection_fallback():
    assert name_smiles("C") == "methane"


def test_component_without_supported_parent_raises_instead_of_methane_fallback():
    mol = Molecule()
    mol.add_atom("H", 0)
    mol.add_atom("H", 1)
    mol.add_bond(0, 1, order=1)

    with pytest.raises(UnnamedAtomError):
        name_component(mol, {0, 1})


def test_component_coverage_audit_rejects_unnamed_atoms():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("O", 1)
    parts = AssemblyParts(parent_length=1, parent_atom_ids={0})

    with pytest.raises(UnnamedAtomError):
        assert_component_fully_named(mol, {0, 1}, parts, "methane")


def test_analysis_carries_name_atom_bindings():
    analysis = analyze_smiles("CCO")
    assembly_steps = [step for step in analysis.decisions if step.decision == "assembled component name"]
    bindings = assembly_steps[-1].data["name_atom_bindings"]
    assert any(binding["role"] == "parent" and binding["atoms"] == [0, 1] for binding in bindings)
    assert any(binding["role"] == "alcohol" and binding["atoms"] == [1, 2] for binding in bindings)


def test_postprocessing_updates_binding_terms():
    bindings = [NameAtomBinding(stage="suffix", role="acid", term="ethanoic acid", atom_ids={0, 1, 2})]

    processed = postprocess_name_atom_bindings(bindings, post_process_name)

    assert processed[0].term == "acetic acid"
    assert processed[0].atom_ids == {0, 1, 2}


def test_stereo_descriptor_boundary_is_inserted_before_substituent_stems():
    assert ensure_stereo_descriptor_boundary("((1R,2R)cyclopropan-2-yl)") == "((1R,2R)-cyclopropan-2-yl)"
    assert ensure_stereo_descriptor_boundary("(1R)-cyclohexyl") == "(1R)-cyclohexyl"


def test_charge_vocabulary_is_registry_backed():
    assert RULES.charges.retained_ionic_n_parents["pyrrolidine"] == "pyrrolidinium"
    assert RULES.charges.saturated_n_ring_ionic_parents[5] == "pyrrolidinium"
    assert RULES.charges.parent_charge_suffixes["N:+"].suffix == "ium"
    assert RULES.charges.replacement_charge_prefixes["aza:+"] == "azonia"
    assert RULES.charges.heteroatom_charge_prefixes["N:+:double"] == "iminio"


def test_namer_rule_sections_are_grouped_by_domain():
    rules = namer_rules()
    groups = rule_groups(rules)
    section_to_group = section_group_map()

    assert not unassigned_sections(rules)
    assert set(section_to_group) == set(rules)
    assert len(section_to_group) == sum(len(spec.sections) for spec in rule_group_specs())
    assert groups["charges"].mapping("parent_charge_suffixes")["N:+"]["suffix"] == "ium"
    assert "functional_groups" in groups["functional_groups"].spec.sections
    assert "postprocess_literal_replacements" in groups["postprocessing"].spec.sections


def test_replacement_parents_are_graph_class_backed():
    assert "replacement_parent_exact_names" not in namer_rules()

    chain = read_smiles("C[Si]#[Si]C")
    assert structural_replacement_parent_name(chain, set(chain.atoms)) == "1,2-dimethyldisilyne"

    oxoacid = read_smiles("OP(=O)(O)O")
    assert structural_replacement_parent_name(oxoacid, set(oxoacid.atoms)) == "phosphoric acid"


def test_halogen_oxoacid_replacement_parents_are_data_backed():
    cases = {
        ("F", 1, 0): "hypofluorous acid",
        ("Cl", 1, 0): "hypochlorous acid",
        ("Cl", 1, 1): "chlorous acid",
        ("Cl", 1, 2): "chloric acid",
        ("Cl", 1, 3): "perchloric acid",
        ("Br", 1, 0): "hypobromous acid",
        ("Br", 1, 1): "bromous acid",
        ("Br", 1, 2): "bromic acid",
        ("Br", 1, 3): "perbromic acid",
        ("I", 1, 0): "hypoiodous acid",
        ("I", 1, 1): "iodous acid",
        ("I", 1, 2): "iodic acid",
        ("I", 1, 3): "periodic acid",
    }

    for (central_symbol, single_o, double_o), expected in cases.items():
        mol = _oxoacid_graph(central_symbol, single_o, double_o)
        assert structural_replacement_parent_name(mol, set(mol.atoms)) == expected


def test_oxoacid_alkyl_esters_are_data_backed():
    cases = {
        ("Cl", 1, 0, 1, 0): "methyl hypochlorite",
        ("Cl", 1, 1, 1, 0): "methyl chlorite",
        ("Cl", 1, 2, 1, 0): "methyl chlorate",
        ("Cl", 1, 3, 1, 0): "methyl perchlorate",
        ("Br", 1, 2, 1, 0): "methyl bromate",
        ("Br", 1, 2, 2, 0): "ethyl bromate",
        ("Br", 1, 2, 3, 0): "propyl bromate",
        ("I", 1, 2, 1, 0): "methyl iodate",
        ("P", 3, 1, 1, 0): "methyl phosphate",
        ("S", 2, 2, 2, 0): "ethyl sulfate",
        ("N", 2, 1, 1, 1): "methyl nitrate",
    }

    for (central_symbol, single_o, double_o, carbon_count, central_charge), expected in cases.items():
        mol = _oxoacid_ester_graph(central_symbol, single_o, double_o, carbon_count, central_charge)
        assert structural_replacement_parent_name(mol, set(mol.atoms)) == expected


def test_charge_normalized_halogen_oxoacid_esters_match_acid_ester_specs():
    acid = _halogen_oxoacid_graph("Br", single_o=1, charged_oxo_o=2, central_charge=2)
    ester = _halogen_oxoacid_ester_graph("Br", charged_oxo_o=2, carbon_count=1, central_charge=2)

    assert structural_replacement_parent_name(acid, set(acid.atoms)) == "bromic acid"
    assert structural_replacement_parent_name(ester, set(ester.atoms)) == "methyl bromate"


def test_oxoacid_esters_use_recursive_front_modifier_namer():
    ester = _halogen_oxoacid_ester_graph("Br", charged_oxo_o=2, carbon_count=1, central_charge=2)
    calls = []

    def branch_namer(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None = None):
        calls.append((start_idx, exclude_atoms, upstream_atom))
        return "(custom modifier)"

    assert structural_replacement_parent_name(ester, set(ester.atoms), branch_namer) == "custom modifier bromate"
    assert calls == [(2, {0, 1, 3, 4}, 1)]


def _oxoacid_graph(central_symbol: str, single_o: int, double_o: int) -> Molecule:
    mol = Molecule()
    mol.add_atom(symbol=central_symbol, idx=0)
    next_idx = 1
    for _ in range(single_o):
        mol.add_atom(symbol="O", idx=next_idx)
        mol.add_bond(0, next_idx, order=1)
        next_idx += 1
    for _ in range(double_o):
        mol.add_atom(symbol="O", idx=next_idx)
        mol.add_bond(0, next_idx, order=2)
        next_idx += 1
    return mol


def _halogen_oxoacid_graph(central_symbol: str, single_o: int, charged_oxo_o: int, central_charge: int) -> Molecule:
    mol = Molecule()
    mol.add_atom(symbol=central_symbol, idx=0, charge=central_charge)
    next_idx = 1
    for _ in range(single_o):
        mol.add_atom(symbol="O", idx=next_idx)
        mol.add_bond(0, next_idx, order=1)
        next_idx += 1
    for _ in range(charged_oxo_o):
        mol.add_atom(symbol="O", idx=next_idx, charge=-1)
        mol.add_bond(0, next_idx, order=1)
        next_idx += 1
    return mol


def _oxoacid_ester_graph(
    central_symbol: str,
    single_o: int,
    double_o: int,
    carbon_count: int,
    central_charge: int = 0,
) -> Molecule:
    mol = Molecule()
    mol.add_atom(symbol=central_symbol, idx=0, charge=central_charge)
    mol.add_atom(symbol="O", idx=1)
    mol.add_bond(0, 1, order=1)
    mol.add_atom(symbol="C", idx=2)
    mol.add_bond(1, 2, order=1)
    next_idx = 3
    previous_carbon = 2
    for _ in range(carbon_count - 1):
        mol.add_atom(symbol="C", idx=next_idx)
        mol.add_bond(previous_carbon, next_idx, order=1)
        previous_carbon = next_idx
        next_idx += 1
    for _ in range(single_o - 1):
        mol.add_atom(symbol="O", idx=next_idx)
        mol.add_bond(0, next_idx, order=1)
        next_idx += 1
    for _ in range(double_o):
        mol.add_atom(symbol="O", idx=next_idx)
        mol.add_bond(0, next_idx, order=2)
        next_idx += 1
    return mol


def _halogen_oxoacid_ester_graph(
    central_symbol: str,
    charged_oxo_o: int,
    carbon_count: int,
    central_charge: int,
) -> Molecule:
    mol = Molecule()
    mol.add_atom(symbol=central_symbol, idx=0, charge=central_charge)
    mol.add_atom(symbol="O", idx=1)
    mol.add_bond(0, 1, order=1)
    mol.add_atom(symbol="C", idx=2)
    mol.add_bond(1, 2, order=1)
    next_idx = 3
    previous_carbon = 2
    for _ in range(carbon_count - 1):
        mol.add_atom(symbol="C", idx=next_idx)
        mol.add_bond(previous_carbon, next_idx, order=1)
        previous_carbon = next_idx
        next_idx += 1
    for _ in range(charged_oxo_o):
        mol.add_atom(symbol="O", idx=next_idx, charge=-1)
        mol.add_bond(0, next_idx, order=1)
        next_idx += 1
    return mol


def test_ring_descriptor_rendering_is_registry_backed():
    assert render_ring_descriptor("spiro", (2, 3)) == "spiro[2.3]"
    assert render_ring_descriptor("bicyclo", (2, 1, 0)) == "bicyclo[2.1.0]"
    assert render_von_baeyer_descriptor(2, "[3.2.2.0^{1,5}]") == "tricyclo[3.2.2.0^{1,5}]"


def test_suffix_stack_places_tested_parent_anion_before_terminal_suffix_stack():
    operation = ParentSuffixOperation(
        key="parent-anion-suffix",
        locants=("5",),
        suffix="ide",
        charge=-1,
        atom_symbols=("C",),
    )

    assert SuffixStack("4-ammonio-1-azacyclohex-3-ene-2,6-dione", operations=[operation]).render() == (
        "4-ammonio-1-azacyclohex-3-ene-5-ide-2,6-dione"
    )
    assert apply_terminal_parent_ide("but-3-ynenitrile", {"2": {"C"}}) == "but-3-ynenitrile"
    assert contains_invalid_locant_ide("but-3-ynenitrile-2-ide")


def test_naming_engine_matches_plain_public_api():
    engine = NamingEngine()

    assert engine.name_smiles("CCO") == name_smiles("CCO")
    assert engine.name_smiles("") == ""
    assert engine.name_smiles_with_trace("CCO")[0] == "ethanol"
    assert engine.analyze_smiles("CCO").name == analyze_smiles("CCO").name


def test_graph_parser_preserves_aromatic_and_hydrogen_metadata_after_kekulization():
    mol = read_smiles("Cn1cc[nH]n1")
    aromatic_n_with_h = [atom.idx for atom in mol if atom.symbol == "N" and atom.is_aromatic and atom.total_h_count > 0]

    assert aromatic_n_with_h


def test_polycycle_numbering_key_uses_explicit_h_metadata_when_enabled():
    mol = read_smiles("c1[nH]ccn1")
    path = [0, 1, 2, 3, 4]

    assert polycycle_numbering_key(mol, path, include_saturated_ring_proxy=True)[1] == (2,)


def test_ring_parent_carries_descriptor_numbering_and_audit_metadata():
    mol = read_smiles("C1CC11CCO1")
    selection = select_principal_parent(mol, [], find_ring_systems(mol), [])

    assert selection is not None
    assert selection.ring_parent is not None
    assert selection.ring_parent.kind == "spiro"
    assert selection.ring_parent.descriptor == "spiro[2.3]"
    assert selection.ring_parent.selected_numbering is not None
    assert selection.ring_parent.selected_numbering.audit_ok
    assert selection.ring_parent.locant_map is not None


def test_display_locants_keep_numeric_order_and_rendered_text():
    locant = as_display_locant(4, "4a")
    numbering = coerce_display_numbering({10: 4, 11: 5}, {10: "4a"})

    assert int(locant) == 4
    assert locant_text(locant) == "4a"
    assert locant_text(numbering[10]) == "4a"
    assert locant_text(numbering[11]) == "5"
    assert sorted([numbering[11], numbering[10]]) == [numbering[10], numbering[11]]
    assert parse_locant(numbering[10]) == (1, 4.0, "a")


def test_functional_groups_carry_metadata_and_graph_bindings():
    mol = read_smiles("CC(=O)O")
    groups = perceive_groups(mol)

    acid = next(group for group in groups if group.key == "carboxylic_acid")
    assert acid.suffix == "oic acid"
    assert acid.prefix == "carboxy"
    assert acid.seniority == 20
    assert acid.atom_ids == {1, 2, 3}
    assert acid.bond_ids == {2, 3}
    assert {binding.role for binding in acid.atom_bindings} == {
        "attachment",
        "characteristic_group",
        "full_group",
    }


def test_custom_perception_detector_extension_point():
    def detector(mol):
        if not mol.atoms:
            return []
        return [PerceivedGroup("fluoro", False, next(iter(mol.atoms)), set())]

    register_group_detector(detector, prepend=True)
    try:
        groups = perceive_groups(read_smiles("C"))
    finally:
        PERCEPTION_DETECTORS.remove(detector)

    custom = next(group for group in groups if group.key == "fluoro")
    assert custom.prefix == "fluoro"
    assert custom.metadata.source == "nomenclature.functional_groups"


def test_functional_group_registry_exposes_derived_families():
    assert "ester" in RULES.functional_groups.keys_with_family("ester_like")
    assert "amide" in RULES.functional_groups.keys_with_family("amide_like")
    assert "ring_nitrile" in RULES.functional_groups.keys_with_family("chain_external_carbonyl")
    assert "hydrazone" in RULES.functional_groups.keys_with_family("hydrazone")
    assert RULES.functional_groups.most_senior(["alcohol", "carboxylic_acid"]).key == "carboxylic_acid"
    assert RULES.functional_groups.direct_subgraph_prefix_for("nitro") == "nitro"
    assert RULES.functional_groups.direct_subgraph_prefix_for("alcohol") is None


def test_legacy_suffix_and_substituent_modules_are_registry_views():
    from structure_to_iupac.rules import substituents, suffixes

    assert suffixes.get("carboxylic_acid").suffix == RULES.functional_groups.get("carboxylic_acid").suffix
    assert substituents.get("nitro").prefix == RULES.functional_groups.get("nitro").prefix
    assert suffixes.get("acid_chloride").render_suffix(3) == "trioyl trichloride"


def test_principal_suffixes_render_programmatically_from_template_positions():
    acid = RULES.functional_groups.get("carboxylic_acid")
    acid_chloride = RULES.functional_groups.get("acid_chloride")
    hydrazone = RULES.functional_groups.get("hydrazone")

    assert acid.multi_suffix is not None
    assert acid.multi_suffix.multiplier_positions == (0,)
    assert acid.suffix_multiplier_positions == (0,)
    assert render_principal_suffix(acid, 1) == "oic acid"
    assert render_principal_suffix(acid, 2) == "dioic acid"
    assert render_principal_suffix(acid, 3) == "trioic acid"

    assert acid_chloride.multi_suffix is not None
    assert acid_chloride.multi_suffix.multiplier_positions == (0, 1)
    assert acid_chloride.suffix_multiplier_positions == (0, 1)
    assert render_principal_suffix(acid_chloride, 2) == "dioyl dichloride"
    assert render_principal_suffix(acid_chloride, 3) == "trioyl trichloride"

    assert hydrazone.suffix_multiplier_positions == (0, 1)
    assert render_principal_suffix(hydrazone, 2) == "dione dihydrazone"
    assert render_principal_suffix(hydrazone, 3) == "trione trihydrazone"


def test_builtin_multi_suffix_rows_are_templates_not_fixed_names():
    groups = namer_rules()["functional_groups"]["values"]
    multi_suffix_rows = [item["multi_suffix"] for item in groups.values() if item.get("multi_suffix") is not None]

    assert multi_suffix_rows
    assert all(isinstance(row, dict) for row in multi_suffix_rows)
    assert all("multiplier_positions" in row for row in multi_suffix_rows)
    assert not any(isinstance(row, str) and row.startswith("di") for row in multi_suffix_rows)


def test_perception_specs_can_extend_group_detection():
    def detector(mol):
        if not mol.atoms:
            return []
        return [PerceivedGroup("chloro", False, next(iter(mol.atoms)), set(), variant="custom", role="prefix")]

    spec = PerceptionDetectorSpec("test.chloro", detector, priority=1, families=("test",))
    register_perception_spec(spec)
    try:
        groups = perceive_groups(read_smiles("C"))
    finally:
        PERCEPTION_SPECS.remove(spec)

    custom = next(group for group in groups if group.variant == "custom")
    assert custom.prefix == "chloro"
    assert custom.role == "prefix"


def test_spiro_marker_is_converted_to_structural_assembly_item():
    parts = AssemblyParts(
        parent_length=3,
        substituents=[SubstituentItem(name="[SPIRO]-2-3-methylaziridine", locants=["1"])],
    )

    spiro_subs = split_spiro_substituents(parts)

    assert spiro_subs == [SpiroAssembly("1", "2", "aziridine", ("3'-methyl",))]
    assert parts.substituents == []


def test_structural_spiro_substituent_bypasses_marker_text():
    parts = AssemblyParts(
        parent_length=3,
        substituents=[
            SubstituentItem(
                name="",
                locants=["1"],
                spiro=SpiroAssembly("1", "2", "aziridine", ("3'-methyl",)),
            )
        ],
    )

    spiro_subs = split_spiro_substituents(parts)

    assert spiro_subs == [SpiroAssembly("1", "2", "aziridine", ("3'-methyl",))]
    assert parts.substituents == []


def test_spiro_side_retained_ionic_ring_locants_are_registry_derived():
    assert extract_spiro_side_prefixes("1-((4-chlorophenyl)methyl)piperidinium") == (
        ["1'-((4-chlorophenyl)methyl)"],
        "piperidin-1-ium",
        (),
    )
    assert _spiro_side_locant(SpiroAssembly("3", "1", "piperidin-1-ium")) == "4"
    assert _spiro_side_locant(SpiroAssembly("3", "1", "pyrrolidin-1-ium")) == "3"
    assert _spiro_side_locant(SpiroAssembly("3", "1", "azetidin-1-ium")) == "3"


def test_charge_and_postprocessing_rules_are_inventory_backed():
    parts = AssemblyParts(
        parent_length=3,
        parent_charges=[ParentChargeItem(atom_id=1, locant="1", symbol="N", charge=1)],
    )
    assert parent_charge_operations(parts)[0].suffix == "ium"
    inventory = postprocessing_rule_inventory()
    assert any(item.owner.startswith("compatibility") for item in inventory)
    assert {item.category for item in inventory} <= {
        "grammar",
        "retained_alias",
        "morphology",
        "opsin_compat",
        "migration",
    }
    assert all(item.reason for item in inventory)


def test_retained_specs_expose_attachment_policy():
    spec = retained_parent_spec("aziridine")
    assert spec is not None
    assert spec.attachment_policy.print_substituent_locant


def test_simple_monocyclic_retained_ring_specs_are_data_driven():
    assert any(spec["name"] == "tetrazole" for spec in RULES.retained.monocycle_specs)

    cases = {
        "c1ccncc1": "pyridine",
        "c1ncc[nH]1": "imidazole",
        "C1CNCCN1": "piperazine",
    }

    for smiles, expected in cases.items():
        mol = read_smiles(smiles)
        ring = find_ring_systems(mol)[0]
        assert get_retained_ring(mol, list(ring.atoms))[0] == expected


def test_fused_and_polycyclic_retained_signatures_are_data_driven():
    spec_names = {spec["name"] for spec in RULES.retained.fused_polycycle_specs}
    assert {"naphthalene", "indole", "adamantane", "cubane", "pyrene"} <= spec_names

    cases = {
        "c1ccc2ccccc2c1": "naphthalene",
        "c1ccc2[nH]ccc2c1": "indole",
        "C1C2CC3CC1CC(C2)C3": "adamantane",
        "C12C3C4C1C5C2C3C45": "cubane",
    }

    for smiles, expected in cases.items():
        mol = read_smiles(smiles)
        ring = find_ring_systems(mol)[0]
        assert get_retained_ring(mol, list(ring.atoms))[0] == expected


def test_tetrazole_substitution_keeps_retained_parent_attachment_explicit():
    assert name_smiles("CC1=NN=NN1") == "5-methyl-1H-tetrazole"
    assert name_smiles("CN1C=NN=N1") == "1-methyl-1H-tetrazole"


def test_retained_heteroaromatic_parent_locants_follow_attachment_equivalence():
    assert name_smiles("Cc1ccccc1") == "methylbenzene"
    assert name_smiles("CN1C=CN=C1") == "1-methyl-1H-imidazole"
    assert name_smiles("CN1C=CC=C1") == "1-methyl-1H-pyrrole"


def test_analyze_smiles_exposes_decision_trace():
    analysis = analyze_smiles("CCO")

    assert analysis.name == "ethanol"
    assert analysis.trace_segments
    phases = [step.phase for step in analysis.decisions]
    assert TracePhase.PARSE in phases
    assert TracePhase.PERCEPTION in phases
    assert TracePhase.PRIORITY in phases
    assert TracePhase.PARENT_SELECTION in phases
    assert TracePhase.NUMBERING in phases
    assert TracePhase.ASSEMBLY in phases


def test_parent_selection_has_named_shape():
    mol = read_smiles("CCO")
    selection = select_principal_parent(
        mol,
        find_all_carbon_paths(mol, exclude_atoms={2}),
        find_ring_systems(mol, exclude_atoms={2}),
        principal_carbons=[1],
    )

    assert isinstance(selection, ParentSelection)
    assert selection.primary_path == [0, 1]
    assert selection.atom_set == {0, 1}
    assert selection.is_ring is False
    assert selection.polycycle_descriptor is None
    assert isinstance(selection.seniority_profile, ParentSeniorityProfile)
    assert selection.seniority_profile.principal_group_count == 1
    assert selection.seniority_profile.parent_atom_count == 2
    assert selection.score_tuple


def test_parent_selection_criteria_are_data_ordered_and_behavior_preserving():
    profile = ParentSeniorityProfile(
        principal_group_count=1,
        contains_principal_group=True,
        senior_element_vector=(7,),
        polycycle_parent=False,
        bicycle_parent=False,
        spiro_parent=False,
        ring_parent=False,
        ring_count=0,
        parent_atom_count=2,
        heteroatom_count=0,
        senior_heteroatom_vector=(),
        senior_heteroatom_count_vector=(0, 0, 0, 0, 0, 0),
        multiple_bond_count=0,
        double_bond_count=0,
        path_tiebreak=(0, 1),
    )

    assert PARENT_SELECTION_CRITERIA == (
        "contains_principal_group",
        "principal_group_count",
        "senior_element_vector",
        "ring_parent",
        "ring_seniority",
        "chain_seniority",
        "multiple_bond_count",
        "double_bond_count",
        "path_tiebreak",
    )
    assert profile.score_tuple() == (-1, -1, (7,), 0, (), (-2, 0, (0, 0, 0, 0, 0, 0)), 0, 0, (0, 1))


def test_parent_seniority_profile_exposes_brief_guide_extension_fields():
    mol = read_smiles("NCCO")
    candidate = ParentCandidate.build(
        [0, 1, 2, 3],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )

    profile = candidate.seniority_profile
    assert profile.senior_element_vector == (1, 5, 7)
    assert profile.heteroatom_count == 2
    assert profile.senior_heteroatom_vector == (1,)
    assert profile.senior_heteroatom_count_vector == (-1, 0, -1, 0, 0, 0)


def test_parent_seniority_criteria_follow_brief_guide_section_6_order():
    assert name_smiles("OCCOCC(=O)O") == "2-(2-hydroxyethoxy)acetic acid"

    mol = Molecule()
    for idx, symbol in {
        0: "C",
        1: "C",
        2: "C",
        3: "N",
        4: "Si",
        5: "O",
        6: "P",
        7: "N",
        8: "N",
        9: "O",
        10: "C",
        11: "C",
        12: "C",
        13: "C",
    }.items():
        mol.add_atom(symbol=symbol, idx=idx)
    for u, v, order in (
        (0, 1, 1),
        (1, 2, 2),
        (10, 11, 1),
        (11, 12, 1),
        (12, 13, 1),
    ):
        mol.add_bond(u, v, order=order)

    no_group = ParentCandidate.build([0, 1], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol)
    one_group = ParentCandidate.build([0, 1], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=1, mol=mol)
    two_groups = ParentCandidate.build([0, 1, 2], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=2, mol=mol)
    n_parent = ParentCandidate.build([3], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol)
    si_parent = ParentCandidate.build([4], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol)
    carbon_ring = ParentCandidate.build([10, 11, 12, 13], is_ring=True, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol, ring_count=1)
    carbon_chain = ParentCandidate.build([10, 11, 12, 13], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol)
    o_ring = ParentCandidate.build([5, 10, 11], is_ring=True, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol, ring_count=1)
    p_ring = ParentCandidate.build([6, 10, 11], is_ring=True, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol, ring_count=1)
    bicycle = ParentCandidate.build([10, 11, 12, 13], is_ring=True, is_bicycle=True, is_spiro=False, is_polycycle=False, xyz=(1, 1, 0), principal_groups_count=0, mol=mol, ring_count=2)
    monocycle = ParentCandidate.build([10, 11, 12, 13], is_ring=True, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol, ring_count=1)
    piperazine_like = ParentCandidate.build([7, 8, 10, 11, 12, 13], is_ring=True, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol, ring_count=1)
    oxazinane_like = ParentCandidate.build([7, 9, 10, 11, 12, 13], is_ring=True, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol, ring_count=1)
    shorter_chain = ParentCandidate.build([10, 11, 12], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol)
    longer_chain = ParentCandidate.build([10, 11, 12, 13], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol)
    saturated_chain = ParentCandidate.build([10, 11, 12], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol)
    unsaturated_chain = ParentCandidate.build([0, 1, 2], is_ring=False, is_bicycle=False, is_spiro=False, is_polycycle=False, xyz=(0, 0, 0), principal_groups_count=0, mol=mol)

    assert min([no_group, one_group], key=lambda candidate: candidate.score_tuple) is one_group
    assert min([one_group, two_groups], key=lambda candidate: candidate.score_tuple) is two_groups
    assert min([si_parent, n_parent], key=lambda candidate: candidate.score_tuple) is n_parent
    assert min([carbon_chain, carbon_ring], key=lambda candidate: candidate.score_tuple) is carbon_ring
    assert min([p_ring, o_ring], key=lambda candidate: candidate.score_tuple) is o_ring
    assert min([monocycle, bicycle], key=lambda candidate: candidate.score_tuple) is bicycle
    assert min([piperazine_like, oxazinane_like], key=lambda candidate: candidate.score_tuple) is oxazinane_like
    assert min([shorter_chain, longer_chain], key=lambda candidate: candidate.score_tuple) is longer_chain
    assert min([saturated_chain, unsaturated_chain], key=lambda candidate: candidate.score_tuple) is unsaturated_chain


def test_numbering_preference_uses_data_ordered_criteria():
    preference = NumberingPreference(
        principal=(2,),
        hetero_by_priority=((1,),),
        indicated_hydrogen=(),
        unsaturation=(3,),
        substituent_and_unsaturation=(3, 4),
        substituent_citation=(4,),
        stereochemistry=(),
    )

    assert NUMBERING_CRITERIA["ring"][0] == "hetero_by_priority"
    assert preference.ring_key()[0] == ((1,),)
    assert NUMBERING_CRITERIA["chain"][0] == "principal"
    assert preference.chain_key()[0] == (2,)


def test_indicated_hydrogen_is_carried_as_hydro_operation():
    mol = read_smiles("C1=NN=NN1")
    parts = AssemblyParts(parent_length=5, is_ring=True, retained_name="tetrazole")
    numbered_path = [0, 1, 2, 3, 4]

    add_indicated_hydrogens(mol, parts, numbered_path, lambda atom_idx: str(numbered_path.index(atom_idx) + 1))

    assert parts.indicated_hydrogens == ["5"]
    assert parts.hydro_operations == [
        HydroOperation(
            key="indicated_hydrogen",
            reason="Retained unsaturated parent requires indicated-hydrogen locant.",
            locants=("5",),
            atom_ids=(4,),
            operation_kind="indicated_hydrogen",
        )
    ]


def test_stereochemistry_audit_checks_emitted_locants_against_graph_metadata():
    mol = read_smiles("F[C@](Cl)(Br)I")
    parts = AssemblyParts(parent_length=1, parent_atom_ids={1}, stereo_features=[("1", "S")])
    parts.parent_atom_ids_by_locant["1"] = 1

    audit = audit_stereochemistry(mol, parts)

    assert audit.ok
    assert audit.checked_features == 1


def test_stereochemistry_audit_checks_bound_substituent_terms():
    mol = read_smiles("F[C@](Cl)(Br)I")
    parts = AssemblyParts(parent_length=1, parent_atom_ids={0})
    parts.name_atom_bindings.append(
        NameAtomBinding(stage="prefix", role="substituent", term="chlorobromoiodomethyl", atom_ids={1, 2, 3, 4})
    )

    audit = audit_stereochemistry(mol, parts)

    assert not audit.ok
    assert "0 R/S descriptors for 1 stereo atoms" in audit.issues[0]


def test_charged_ammonio_substituent_keeps_all_n_ligands_explicit():
    assert (
        name_smiles("CCCC1CCC(CC1)[NH+](C)Cc2ccc3c(c2)oc(=O)o3")
        == "3-(((4-propylcyclohexyl)(methyl)ammonio)methyl)-7,9-dioxabicyclo[4.3.0]nona-1(6),2,4-trien-8-one"
    )


def test_zinc_multi_locant_cation_suffixes_render_with_multiplier():
    assert (
        name_smiles("CC(C)(C[NH+]1CCC[C@@]2(C1)CC[NH2+]C2)COC")
        == "(5R)-7-(2-(methoxymethyl)-2-methylpropyl)-2,7-diazaspiro[4.5]decan-2,7-diium"
    )


def test_parent_candidate_score_prefers_principal_coverage_then_length():
    short = ParentCandidate.build(
        [0],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=1,
    )
    long = ParentCandidate.build(
        [0, 1],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=1,
    )

    assert min([short, long], key=lambda candidate: candidate.score_tuple) is long
    assert long.seniority_profile.parent_atom_count == 2


def test_parent_pipeline_uses_intent_to_build_component_parts():
    mol = read_smiles("CCO")
    selection = select_principal_parent(
        mol,
        find_all_carbon_paths(mol, exclude_atoms={2}),
        find_ring_systems(mol, exclude_atoms={2}),
        principal_carbons=[1],
    )

    plan = build_parent_assembly_plan(
        mol,
        selection,
        NamingIntent.component([1]),
        {},
        None,
        None,
    )

    assert plan.numbered_path == [1, 0]
    assert plan.get_loc(1) == "1"
    assert plan.parts.parent_atom_ids == {0, 1}
    assert plan.parts.is_substituent is False


def test_heteroatom_substituent_specs_are_data_shaped():
    assert unsubstituted_prefix("O") == "hydroxy"
    assert ligand_prefix("O", "methyl") == "methoxy"
    assert unsubstituted_prefix("S", 2) == "sulfonyl"


def test_ring_system_fragment_maps_local_and_global_atoms():
    mol = read_smiles("C1CCCCC1O")
    fragment = ring_system_fragment(mol, {0, 1, 2, 3, 4, 5})

    assert set(fragment.old_to_new) == {0, 1, 2, 3, 4, 5}
    assert set(fragment.new_to_old.values()) == {0, 1, 2, 3, 4, 5}
    assert fragment.global_atom(fragment.local_atom(3)) == 3
    assert fragment.global_numbering({fragment.local_atom(0): 1}) == {0: 1}
    assert len(fragment.fragment.atoms) == 6


def test_public_api_golden_names():
    cases = {
        "": "",
        "C": "methane",
        "N": "azane",
        "O": "oxidane",
        "CC": "ethane",
        "CCC": "propane",
        "C=C": "eth-1-ene",
        "C#C": "eth-1-yne",
        "CCO": "ethanol",
        "CC(=O)O": "acetic acid",
        "C1CCCCC1": "cyclohexane",
        "[Na+].[Cl-]": "sodium chloride",
        "c1ccccc1O": "benzen-1-ol",
        "CC(=O)Oc1ccccc1C(=O)O": "2-(acetoxy)benzoic acid",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_use_parseable_spiro_and_formamido_forms():
    cases = {
        "CN1CC11C2C3CC2C13": "1'-methylspiro[tricyclo[2.2.0.0^{2,5}]hexane-6,2'-aziridine]",
        "CC1CC11CC2CCC12": "2'-methylspiro[bicyclo[2.2.0]hexane-2,1'-cyclopropane]",
        "CC1NC11CC2OC12C": "1-methyl-3'-methylspiro[5-oxabicyclo[2.1.0]pentane-2,2'-aziridine]",
        "COC(=O)CCNC=O": "methyl 3-(formamido)propanoate",
        "N=COC(=O)CNC=O": "iminomethyl 2-(formamido)acetate",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_preserve_positive_nitrogen_charge():
    cases = {
        "[O-]C(=O)C[NH+]1CCC1": "2-(azetidinium-1-yl)acetate",
        "CCN(C1CCC1)C(=O)C[C@H]1CSCC[NH2+]1": "N-cyclobutyl-N-ethyl-2-((3S)-thiomorpholin-4-ium-3-yl)acetamide",
        "CC(CCC[NH3+])C([O-])=O": "5-ammonio-2-methylpentanoate",
        "CC([NH3+])C(C)(N)C([O-])=O": "2-amino-3-ammonio-2-methylbutanoate",
        "OC1C[NH2+]C1C([O-])=O": "3-hydroxyazetidinium-2-carboxylate",
        "C[NH+](C)C(C#C)C([O-])=O": "2-(dimethylammonio)but-3-ynoate",
        "[O-]C(=O)C1[NH2+]CC2CC12": "3-azoniabicyclo[3.1.0]hexane-2-carboxylate",
        "CC(C)(C)/C=C/C(=O)N1CCC[C@@H]2[C@H]1C[NH2+]C2": "(2E)-1-((1S,6S)-2,8-diazabicyclo[4.3.0]nonan-8-ium-2-yl)-4,4-dimethylpent-2-en-1-one",
        "C[C@H]1C[NH+](CCN1c2[nH]c3ccccc3n2)C": "2-((2S)-2,4-dimethylpiperazin-4-ium-1-yl)-3H-benzimidazole",
        "Cc1cn2c(cccc2[nH+]1)c3cccc(c3F)C[NH+]4CCCCC4": "2-(2-fluoro-3-(piperidinium-1-ylmethyl)phenyl)-8-methyl-1,7-diazabicyclo[4.3.0]nona-2,4,6,8-tetraen-7-ium",
        "Cc1cn2c(cccc2[nH+]1)c3ccc(cc3F)N4CCCCC4": "2-(2-fluoro-4-(piperidin-1-yl)phenyl)-8-methyl-1,7-diazabicyclo[4.3.0]nona-2,4,6,8-tetraen-7-ium",
        "C[C@@H]1CC[C@H](C[NH2+]1)N2C[C@@H](C[C@H]2C)F": "(2R,5R)-5-((2R,4R)-4-fluoro-2-methylpyrrolidin-1-yl)-2-methylpiperidinium",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_parent_charge_layer_uses_structured_sites_and_charge_filters():
    mol = read_smiles("[O-]C(=O)C[NH+]1CCC1")
    numbering = {3: "1", 4: "2", 5: "3", 6: "4"}
    sites = parent_charge_sites(mol, [3, 4, 5, 6], lambda atom_idx: numbering[atom_idx])

    assert [(site.symbol, site.charge, site.locant) for site in sites] == [("N", 1, "2")]
    assert (
        apply_parent_charge_names(
            "azetidin-2-yl",
            mol,
            [3, 4, 5, 6],
            lambda atom_idx: numbering[atom_idx],
            allow_retained_stem_inference=True,
            charge_signs={1},
        )
        == "azetidinium-2-yl"
    )
    assert (
        apply_parent_charge_names(
            "azetidin-2-yl",
            mol,
            [3, 4, 5, 6],
            lambda atom_idx: numbering[atom_idx],
            allow_retained_stem_inference=True,
            charge_signs={-1},
        )
        == "azetidin-2-yl"
    )


def test_cationic_parent_suffixes_are_created_by_assembly_parts():
    retained_parts = AssemblyParts(
        parent_length=4,
        is_ring=True,
        is_substituent=True,
        attachment_locant=1,
        parent_charges=[ParentChargeItem(locant="1", symbol="N", charge=1)],
        a_prefixes=[SubstituentItem(name="aza", locants=["1"])],
    )
    fused_parts = AssemblyParts(
        parent_length=9,
        is_bicycle=True,
        bicycle_xyz=(4, 3, 0),
        a_prefixes=[
            SubstituentItem(name="aza", locants=["1"]),
            SubstituentItem(name="aza", locants=["7"]),
        ],
        unsaturations=[UnsaturationItem(bond_key="double", locants=["2", "4", "6", "8"])],
        parent_charges=[ParentChargeItem(locant="7", symbol="N", charge=1)],
    )

    assert assemble_name(retained_parts) == "azetidinium-1-yl"
    assert assemble_name(fused_parts) == "1,7-diazabicyclo[4.3.0]nona-2,4,6,8-tetraen-7-ium"


def test_multiple_cationic_parent_suffixes_use_multiplicative_suffixes():
    parent_parts = AssemblyParts(
        parent_length=9,
        is_bicycle=True,
        bicycle_xyz=(3, 3, 0),
        a_prefixes=[
            SubstituentItem(name="aza", locants=["3"]),
            SubstituentItem(name="aza", locants=["7"]),
        ],
        parent_charges=[
            ParentChargeItem(locant="3", symbol="N", charge=1),
            ParentChargeItem(locant="7", symbol="N", charge=1),
        ],
    )
    substituent_parts = AssemblyParts(
        parent_length=9,
        is_bicycle=True,
        is_substituent=True,
        attachment_locant=3,
        bicycle_xyz=(3, 3, 0),
        a_prefixes=[
            SubstituentItem(name="aza", locants=["3"]),
            SubstituentItem(name="aza", locants=["7"]),
        ],
        parent_charges=[
            ParentChargeItem(locant="3", symbol="N", charge=1),
            ParentChargeItem(locant="7", symbol="N", charge=1),
        ],
    )

    assert assemble_name(parent_parts) == "3,7-diazabicyclo[3.3.0]nonan-3,7-diium"
    assert assemble_name(substituent_parts) == "3,7-diazabicyclo[3.3.0]nonan-3,7-diium-3-yl"


def test_ambiguous_ring_connection_stems_keep_attachment_locants():
    cases = {
        "pyrazole": "pyrazol-1-yl",
        "piperazine": "piperazin-1-yl",
        "aziridine": "aziridin-1-yl",
        "benzene": "phenyl",
    }

    for retained_name, expected in cases.items():
        parts = AssemblyParts(
            parent_length=6 if retained_name in {"piperazine", "benzene"} else 5,
            is_ring=True,
            is_substituent=True,
            retained_name=retained_name,
            attachment_locant=1,
        )
        if retained_name == "aziridine":
            parts.parent_length = 3
        assert assemble_name(parts) == expected


def test_spiro_substituent_radicals_keep_attachment_locants():
    cases = {
        "OCC12CC(C1)C21CN1": "1-(spiro[bicyclo[1.1.1]pentane-2,2'-aziridine]-1-yl)methanol",
        "OCC12CC1C1(CN1)C2": "1-(spiro[bicyclo[2.1.0]pentane-3,2'-aziridine]-1-yl)methanol",
        "OCC1CC11C2CC1C2": "1-(spiro[cyclopropane-2,2'-bicyclo[1.1.1]pentane]-1-yl)methanol",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_spiro_side_parents_are_normalized_without_nested_or_parenthesized_polycycles():
    prefixes, parent, suffixes = extract_spiro_side_prefixes("1-methyl-bicyclo[1.1.1]pentane")
    assert prefixes == ["1'-methyl"]
    assert parent == "bicyclo[1.1.1]pentane"
    assert suffixes == ()

    core, _ = format_spiro_core(
        "cyclopropan",
        "",
        "e",
        [
            SpiroAssembly("2", "2", "bicyclo[1.1.1]pentane", ()),
            SpiroAssembly("3", "1", "cyclopropane", ()),
        ],
    )
    assert "spiro[spiro[" not in core
    assert "(bicyclo" not in core


def test_spiro_side_heteroaromatic_branch_uses_graph_numbered_side_ring():
    mol = Molecule()
    for idx, symbol in {
        0: "C",
        1: "C",
        2: "C",
        3: "C",
        4: "N",
        5: "C",
        6: "O",
        7: "N",
        8: "C",
    }.items():
        mol.add_atom(symbol=symbol, idx=idx, is_aromatic=idx in {3, 4, 5, 6, 7})
    for u, v, order in (
        (0, 1, 1),
        (1, 2, 1),
        (2, 0, 1),
        (1, 5, 1),
        (3, 4, 1),
        (4, 5, 2),
        (5, 6, 1),
        (6, 7, 1),
        (7, 3, 2),
        (3, 8, 1),
    ):
        mol.add_bond(u, v, order=order)

    spiro = _spiro_subgraph_assembly(mol, 0, set(mol.atoms))

    assert spiro.side_locant == "1"
    assert spiro.side_parent_name == "cyclopropane"
    assert spiro.side_prefixes == ("2'-(3-methyl-1,2,4-oxadiazol-5-yl)",)


def test_spiro_side_hantzsch_widman_branch_requires_aromatic_ring_metadata():
    mol = Molecule()
    for idx, symbol in {
        0: "C",
        1: "C",
        2: "C",
        3: "C",
        4: "N",
        5: "C",
        6: "O",
        7: "N",
        8: "C",
    }.items():
        mol.add_atom(symbol=symbol, idx=idx, is_aromatic=False)
    for u, v, order in (
        (0, 1, 1),
        (1, 2, 1),
        (2, 0, 1),
        (1, 5, 1),
        (3, 4, 1),
        (4, 5, 2),
        (5, 6, 1),
        (6, 7, 1),
        (7, 3, 2),
        (3, 8, 1),
    ):
        mol.add_bond(u, v, order=order)

    spiro = _spiro_subgraph_assembly(mol, 0, set(mol.atoms))

    assert spiro.side_parent_name != "cyclopropane"
    assert not any("oxadiazol" in prefix for prefix in spiro.side_prefixes)


def test_mixed_spiro_bicyclo_side_suffixes_and_replacement_locants_are_component_scoped():
    cases = {
        "OC1CC11C2CC1(O)C2": "spiro[bicyclo[1.1.1]pentane-2,1'-cyclopropane]-2'-ol-1-ol",
        "OC1CC11C2CC1C2=O": "spiro[bicyclo[1.1.1]pentane-4,1'-cyclopropane]-2'-ol-2-one",
        "CC12CC(O1)C21CC1O": "1'-methyl-2'-oxaspiro[cyclopropane-2,4'-bicyclo[1.1.1]pentane]-1-ol",
        "CC12CN(C1)C21CC1O": "3'-methyl-1'-azaspiro[cyclopropane-2,2'-bicyclo[1.1.1]pentane]-1-ol",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_polycycle_topology_classifies_linear_dispiro_and_rejects_mixed_spiro():
    linear = read_smiles("C1CC11CC2(CC2)C1")
    linear_system = next(system for system in find_ring_systems(linear) if system.is_polycycle)
    linear_topology = ring_system_topology(linear, linear_system.atoms)
    proof = linear_dispiro_proof(linear_topology.atoms, linear_topology.edges)

    assert linear_topology.classification == "linear_dispiro"
    assert proof is not None
    assert proof.descriptor == "dispiro[2.1.2.1]"
    assert proof.atom_count == len(linear_system.atoms)

    mixed = read_smiles("C1C2C1C1(CC1)C21CC1")
    mixed_topology = ring_system_topology(mixed, set(mixed.atoms))

    assert mixed_topology.classification != "linear_dispiro"
    assert linear_dispiro_proof(mixed_topology.atoms, mixed_topology.edges) is None


def test_polycycle_topology_proves_monospiro_and_bicyclo_descriptors():
    spiro_mol = read_smiles("C1CC11CC1")
    spiro_system = next(system for system in find_ring_systems(spiro_mol) if system.is_spiro)
    spiro_topology = ring_system_topology(spiro_mol, spiro_system.atoms)
    spiro = monospiro_proof(spiro_topology.atoms, spiro_topology.edges)

    assert spiro_topology.classification == "monospiro"
    assert spiro is not None
    assert spiro.descriptor == "spiro[2.2]"
    assert spiro.atom_count == len(spiro_system.atoms)

    bicyclo_mol = read_smiles("C1CC2CC1C2")
    bicyclo_system = next(system for system in find_ring_systems(bicyclo_mol) if system.is_bicycle)
    bicyclo_topology = ring_system_topology(bicyclo_mol, bicyclo_system.atoms)
    bicyclo = bicyclo_proof(bicyclo_topology.atoms, bicyclo_topology.edges)

    assert bicyclo_topology.classification == "bicyclic"
    assert bicyclo is not None
    assert bicyclo.descriptor == "bicyclo[2.1.1]"
    assert bicyclo.atom_count == len(bicyclo_system.atoms)


def test_simple_spiro_numbering_map_keeps_smaller_ring_first_and_audits_edges():
    mol = read_smiles("C1CC11CCO1")
    system = next(system for system in find_ring_systems(mol) if system.is_spiro)
    topology = ring_system_topology(mol, system.atoms)
    proof = monospiro_proof(topology.atoms, topology.edges)

    assert proof is not None
    assert proof.descriptor_numbers == (2, 3)
    assert system.paths == [list(path) for path in proof.numbering_paths]
    assert all(path[2] == proof.spiro_atom for path in proof.numbering_paths)

    numberings = [
        build_ring_numbering("spiro", proof.descriptor_numbers, path, topology.edges, mol)
        for path in proof.numbering_paths
    ]
    assert all(numbering.audit_ok for numbering in numberings)
    assert all(numbering.spiro_locants == (3,) for numbering in numberings)
    assert all(numbering.atom_to_locant[5] in {4, 6} for numbering in numberings)
    assert {numbering.atom_to_locant[5] for numbering in numberings} == {4, 6}
    assert all(numbering.atom_symbols_by_locant[numbering.atom_to_locant[5]] == "O" for numbering in numberings)
    assert all(set(numbering.bond_orders_by_locants.values()) == {1} for numbering in numberings)
    assert name_smiles("C1CC11CCO1") == "4-oxaspiro[2.3]hexane"
    assert name_smiles("CC1CC2(CC2)O1") == "5-methyl-4-oxaspiro[2.3]hexane"
    assert name_smiles("OC1CC11CC1") == "spiro[2.2]pentan-1-ol"
    assert name_smiles("O=C1CC11CC1") == "spiro[2.2]pentan-1-one"


def test_simple_bicyclo_numbering_map_matches_bridge_descriptor_and_attachment_locant():
    mol = read_smiles("CC1C2CCC12")
    system = next(system for system in find_ring_systems(mol) if system.is_bicycle)
    topology = ring_system_topology(mol, system.atoms)
    proof = bicyclo_proof(topology.atoms, topology.edges)

    assert proof is not None
    assert proof.descriptor_numbers == (2, 1, 0)
    assert system.paths == [list(path) for path in proof.numbering_paths]

    numberings = [
        build_ring_numbering("bicyclo", proof.descriptor_numbers, path, topology.edges, mol, substituent_attachment_atoms={1})
        for path in proof.numbering_paths
    ]
    assert all(numbering.audit_ok for numbering in numberings)
    assert all(numbering.bridgehead_locants == (1, 4) for numbering in numberings)
    assert all(numbering.atom_to_locant[1] == 5 for numbering in numberings)
    assert all(numbering.substituent_attachment_locants == (5,) for numbering in numberings)
    assert all(set(numbering.bond_orders_by_locants.values()) == {1} for numbering in numberings)
    assert name_smiles("CC1C2CCC12") == "5-methylbicyclo[2.1.0]pentane"
    assert name_smiles("OC1CC2CC1C2") == "bicyclo[2.1.1]hexan-2-ol"
    assert name_smiles("O=C1CC2CC1C2") == "bicyclo[2.1.1]hexan-2-one"


def test_linear_polyspiro_ring_systems_use_dispiro_parent_not_partial_spiro():
    cases = {
        "C1CC11CCC11CC1": "dispiro[2.0.2.2]octane",
        "C1CC11COC11CC1": "7-oxadispiro[2.0.2.2]octane",
        "C1CC2(C1)OC21CCC1": "9-oxadispiro[3.0.3.1]nonane",
        "C1OC11CC2(CCC2)C1": "1-oxadispiro[2.1.3.1]nonane",
        "C1CC11CC2(CC2)C1": "dispiro[2.1.2.1]octane",
        "C1=CC2(CN2)CC12CN2": "1,7-diazadispiro[2.2.2.1]non-4-ene",
    }

    for smiles, expected in cases.items():
        generated = name_smiles(smiles)
        assert generated == expected
        assert generated != "spiro[2.3]hexane"
        assert "methyl" not in generated
        assert "ethenyl" not in generated


def test_dispiro_numbering_map_audits_direct_and_non_direct_middle_segments():
    direct = read_smiles("C1CC11COC11CC1")
    direct_system = next(system for system in find_ring_systems(direct) if system.is_polycycle)
    direct_topology = ring_system_topology(direct, direct_system.atoms)
    direct_numberings = [
        build_ring_numbering("dispiro", (2, 0, 2, 2), path, direct_topology.edges, direct)
        for path in direct_system.paths
    ]

    assert direct_system.polycycle_descriptor == "dispiro[2.0.2.2]"
    assert all(numbering.audit_ok for numbering in direct_numberings)
    assert {numbering.atom_to_locant[4] for numbering in direct_numberings} == {7}
    assert all(numbering.atom_symbols_by_locant[numbering.atom_to_locant[4]] == "O" for numbering in direct_numberings)

    non_direct = read_smiles("C1OC11CC2(CCC2)C1")
    non_direct_system = next(system for system in find_ring_systems(non_direct) if system.is_polycycle)
    non_direct_topology = ring_system_topology(non_direct, non_direct_system.atoms)
    non_direct_numberings = [
        build_ring_numbering("dispiro", (2, 1, 3, 1), path, non_direct_topology.edges, non_direct)
        for path in non_direct_system.paths
    ]

    assert non_direct_system.polycycle_descriptor == "dispiro[2.1.3.1]"
    assert all(numbering.audit_ok for numbering in non_direct_numberings)
    assert {numbering.atom_to_locant[1] for numbering in non_direct_numberings} == {1, 2}
    assert all(set(numbering.bond_orders_by_locants.values()) == {1} for numbering in non_direct_numberings)


def test_split_polyspiro_oxirane_case_keeps_both_spiro_side_rings():
    generated = name_smiles("O1CC11C2CC2C11OC1")

    assert generated == "dispiro[oxirane-2,2'-bicyclo[2.1.0]pentane-3',2''-oxirane]"


def test_complex_spiro_fused_systems_do_not_use_linear_dispiro_renderer():
    generated = name_smiles("C1C2C1C1(CC1)C21CC1")

    assert generated == "dispiro[cyclopropane-1,2'-bicyclo[2.1.0]pentane-3',1''-cyclopropane]"
    assert generated != "dispiro[2.0.2.3]nonane"


def test_pyopsin_regression_names_preserve_terminal_olate_charge():
    cases = {
        "[NH3+]CC1=C([O-])OC=N1": "4-(ammoniomethyl)oxazol-5-olate",
        "C[NH2+]CC1=C([O-])OC=C1": "3-((methylammonio)methyl)furan-2-olate",
        "[NH3+]CC1=C([O-])OC=C1": "3-(ammoniomethyl)furan-2-olate",
        "C[NH2+]CC1=NOC([O-])=C1": "3-((methylammonio)methyl)isoxazol-5-olate",
        "[NH3+]CC1=CC(O)=C([O-])O1": "5-(ammoniomethyl)-3-hydroxyfuran-2-olate",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_preserve_retained_ring_anions():
    cases = {
        "[NH3+]CCOC1=NC=N[N-]1": "2-((1,2,4-triazol-1-ide-5-yl)oxy)ethan-1-aminium",
        "[NH3+]CC#CC1=NC=N[N-]1": "3-(1,2,4-triazol-1-ide-5-yl)prop-2-yn-1-aminium",
        "CC([NH3+])C1=CN=N[N-]1": "1-(1,2,3-triazol-1-ide-5-yl)ethan-1-aminium",
        "[NH3+]CC1=N[N-]N=N1": "1-(tetrazol-3-ide-5-yl)methanaminium",
        "[NH3+]C[C-]1OC(=O)C=C1": "5-(ammoniomethyl)-2-oxo-1-oxacyclopent-3-en-5-ide",
        "[NH3+]CCC1=N[N-]C(=O)O1": "5-(2-ammonioethyl)-2-oxo-1-oxa-3,4-diazacyclopent-4-en-3-ide",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_preserve_retained_ring_cations():
    cases = {
        "CC(=C(C)C(=O)NCCc1[nH+]ccn1C)C": "2,3-dimethyl-N-(2-(1-methyl-imidazol-3-ium-2-yl)ethyl)but-2-enamide",
        "CC[C@@H](C(=O)CC)Oc1[nH]c(c[nH+]1)C(=O)OC": "methyl 2-(((3S)-4-oxohexan-3-yl)oxy)-imidazol-3-ium-5-carboxylate",
        "Cn1c(ccn1)C[NH+]2CCc3c(cc[nH]c3=O)C2": "8-((1-methyl-1H-pyrazol-5-yl)methyl)-3,8-diazabicyclo[4.4.0]deca-1(6),4-dien-8-ium-2-one",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_use_substituted_carbamoyl_prefixes():
    cases = {
        "CNC(=O)NC(C)=O": "N-(methylcarbamoyl)acetamide",
        "CC(C)(C(=O)N(C)C1CCC1)NC(=O)OC": "methyl ((2-((cyclobutyl)(methyl)carbamoyl)propan-2-yl)amino)formate",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_preserve_anionic_carbamoylamino_and_thio_charge():
    cases = {
        "CC(C)CSCCCOC(=O)[N-]C(=O)N": "3-((2-methylpropyl)sulfanyl)propyl (carbamoylazanidyl)formate",
        "CCOC(=O)[N-]C(=S)NC(C)(C)C1CC1": "ethyl (((2-cyclopropylpropan-2-yl)carbamothioyl)azanidyl)formate",
        "CCOC(=O)[N-]C(=S)N(C)[C@H]1CCCOC1": "ethyl ((((3S)-oxan-3-yl)(methyl)carbamothioyl)azanidyl)formate",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_keep_formate_esters_principal():
    cases = {
        "O=COCC#N": "2-nitriloethyl formate",
        "O=COCCC#N": "3-nitrilopropyl formate",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_preserve_carbanion_suffix_locants():
    cases = {
        "C[NH2+]CC(=O)[CH-]C=O": "4-(methylammonio)-3-oxobutan-2-ide-1-al",
        "NC=[NH+][C-](C#N)C#N": "2-(aminomethylideneammonio)propane-2-ide-1,3-dinitrile",
        "[NH2+]=C1O[C-](C=C1)C#N": "5-iminio-1-oxacyclopent-3-ene-2-ide-2-carbonitrile",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_pyopsin_regression_names_preserve_zwitterionic_parent_suffix_order():
    cases = {
        "[NH3+]C1=CC(=O)NC(=O)[CH-]1": "4-ammonio-1-azacyclohex-3-ene-5-ide-2,6-dione",
        "NC1=NC(N)=[NH+][N-]C1=N": "6-imino-1,2,4-triazacyclohexa-2,4-dien-2-ium-1-ide-3,5-diamine",
        "[NH3+][C-]1C=CC2=C1N=NO2": "2-oxa-3,4-diazabicyclo[3.3.0]octa-1(5),3,7-trien-6-ide-6-aminium",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_unclassified_anions_do_not_emit_terminal_locant_ide():
    cases = {
        "NC1=[NH+]C(=N)N=C[N-]1": "6-imino-1,3,5-triazacyclohexa-1,4-dien-1-ium-2-amine",
        "NC(=[NH2+])[C-](C#C)C#N": "2-((amino)(iminio)methyl)but-3-ynenitrile",
        "O=C[C-]1C2C[NH2+]C12C=O": "2-azoniabicyclo[2.1.0]pentane-1,5-dicarbaldehyde",
        "[NH3+]C1CC(=O)[N-]C1C#N": "3-ammonio-5-oxopyrrolidine-2-carbonitrile",
    }

    for smiles, expected in cases.items():
        name = name_smiles(smiles)
        assert name == expected
        assert not contains_invalid_locant_ide(name)


def test_pyopsin_regression_names_preserve_cationic_imidamide_connectivity():
    cases = {
        "CNC(N)=[NH+]CC([O-])=O": "2-(N-((amino)(methylamino)methylidene)ammonio)acetate",
        "COC(N)=[NH+]CC([O-])=O": "2-(N-((amino)(methoxy)methylidene)ammonio)acetate",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_invalid_ide_morphology_is_normalized():
    assert (
        apply_retained_parent_ide("5-(ammoniomethyl)imidazolidine-2,4-dione", "imidazolidine", {"1": {"C"}})
        == "5-(ammoniomethyl)imidazolidin-1-ide-2,4-dione"
    )
    assert (
        apply_retained_parent_ide(
            "5-(ammoniomethyl)-1,2,3-triazole-4-carbonitrile", "1,2,3-triazole", {"1": {"C"}}
        )
        == "5-(ammoniomethyl)-1,2,3-triazol-1-ide-4-carbonitrile"
    )


def test_multiplied_formylamino_names_use_n_locants_not_diformamido():
    cases = {
        "O=CN(C=O)CC(N)=O": "2-(N,N-diformylamino)acetamide",
        "COC(=O)N(C=O)C=O": "methyl (N,N-diformylamino)formate",
        "O=CN(C=O)C(=O)OC": "methyl (N,N-diformylamino)formate",
    }

    for smiles, expected in cases.items():
        generated = name_smiles(smiles)
        assert generated == expected
        assert "diformamido" not in generated


def test_pyopsin_regression_names_preserve_cationic_imino_charge():
    cases = {
        "NC(=[NH2+])C1=C(O)N=N[N-]1": "5-((amino)(iminio)methyl)-1,2,3-triazol-1-ide-4-ol",
        "CNC(=[NH2+])C(CO)=N[O-]": "3-iminio-3-(methylamino)-2-(oxidoimino)propan-1-ol",
        "NC(=[NH2+])C(CC#C)=N[O-]": "1-iminio-2-(oxidoimino)pent-4-yn-1-amine",
        "C[NH+]=C(N)C(CO)=N[O-]": "3-amino-3-(methyliminio)-2-(oxidoimino)propan-1-ol",
        "CC(CC([O-])=O)NC=[NH2+]": "3-(iminiomethylamino)butanoate",
        "CN(CC([O-])=O)C=[NH2+]": "2-((iminiomethyl)(methyl)amino)acetate",
        "[O-]C(=O)C(=[NH2+])NC1CC1": "2-(cyclopropylamino)-2-iminioacetate",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_charged_fused_heteroaromatic_bicycles_spell_nitrogen_zwitterion():
    cases = {
        "[N-]1[NH+]=CC=C2C=CN=C12": "2,3,9-triazabicyclo[4.3.0]nona-1(9),3,5,7-tetraen-2-ide-3-ium",
        "[N-]1[NH+]=CC=C2N=CC=C12": "2,3,7-triazabicyclo[4.3.0]nona-1(9),3,5,7-tetraen-2-ide-3-ium",
        "[N-]1[NH+]=CC=C2N=CN=C12": "2,3,7,9-tetraazabicyclo[4.3.0]nona-1(9),3,5,7-tetraen-2-ide-3-ium",
        "[N-]1[NH+]=CN=C2C=CN=C12": "2,3,5,9-tetraazabicyclo[4.3.0]nona-1(9),3,5,7-tetraen-2-ide-3-ium",
        "[N-]1[NH+]=CN=C2N=CN=C12": "2,3,5,7,9-pentaazabicyclo[4.3.0]nona-1(9),3,5,7-tetraen-2-ide-3-ium",
        "[N-]1[NH+]=NC=C2N=CN=C12": "2,3,4,7,9-pentaazabicyclo[4.3.0]nona-1(9),3,5,7-tetraen-2-ide-3-ium",
    }

    for smiles, expected in cases.items():
        generated = name_smiles(smiles)
        assert generated == expected
        assert not contains_invalid_locant_ide(generated)


def test_ring_parent_nitrogen_zwitterion_stack_is_graph_gated_not_name_literal():
    mol = read_smiles("[N-]1[NH+]=CC=C2C=CN=C12")
    numbered_path = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    locants = {atom_idx: str(idx + 1) for idx, atom_idx in enumerate(numbered_path)}
    generic_name = "1,2,8-triazatricyclo[4.3.0.0]nona-1,3,5,7-tetraen-2-ium"

    updated = apply_ring_parent_nitrogen_zwitterion_stack(
        generic_name,
        mol,
        numbered_path,
        lambda atom_idx: locants[atom_idx],
    )

    assert updated == "1,2,8-triazatricyclo[4.3.0.0]nona-1,3,5,7-tetraen-1-ide-2-ium"


def test_ring_parent_nitrogen_zwitterion_stack_rejects_acyclic_ium_names():
    mol = read_smiles("[NH+]=C[N-]")
    numbered_path = [0, 1, 2]

    updated = apply_ring_parent_nitrogen_zwitterion_stack(
        "1,3-diaza-prop-1-en-1-ium",
        mol,
        numbered_path,
        lambda atom_idx: str(atom_idx + 1),
    )

    assert updated == "1,3-diaza-prop-1-en-1-ium"


def test_connection_boundary_regression_names_keep_unambiguous_attachment():
    cases = {
        "CCN([C@H](C)CO)S(=O)(=O)c1ccccc1": "(2R)-2-(N-ethylbenzenesulfonamido)propan-1-ol",
        "Cc1[nH+]c(cn1Cc2cc(cnc2)F)C(=O)OC": "methyl 1-((5-fluoropyridin-3-yl)methyl)-2-methyl-imidazol-3-ium-4-carboxylate",
        "COc1c(cccc1)C2(CC2)C(=O)O[C@H]3C[C@H](C3)c4ccccc4": "3-phenylcyclobutyl 1-(2-methoxyphenyl)cyclopropanecarboxylate",
        "CO[C@H]1C[C@H](C1)OC(=O)c2n(ncc2)C(F)F": "3-methoxycyclobutyl 1-(difluoromethyl)-1H-pyrazole-5-carboxylate",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_connection_boundary_postprocessing_is_data_driven_by_parent_stem():
    assert (
        apply_connection_boundary_postprocessing("ethyl 1-(methylsulfonyl)piperidine-4-carboxylate")
        == "ethyl 1-(methylsulfonyl)piperidine-4-carboxylate"
    )
    assert apply_connection_boundary_postprocessing("ethyl 1-(chloro)acetate") == "ethyl (chloro)acetate"
    assert (
        apply_connection_boundary_postprocessing("(2R)-2-((ethyl)benzenesulfonamido)propan-1-ol")
        == "(2R)-2-(N-ethylbenzenesulfonamido)propan-1-ol"
    )


def test_anionic_ketone_parent_names_keep_parent_descriptor_intact():
    cases = {
        "O=C1[CH-][NH+]2CCC2=C1": "3-oxo-1-azoniabicyclo[3.2.0]hept-4-en-2-ide",
        "O=C1C=C[NH+]2CC[C-]12": "4-oxo-1-azoniabicyclo[3.2.0]hept-2-en-5-ide",
        "O=C1[CH-]NC2=C1C[NH2+]C2": "4-oxo-2,7-diazabicyclo[3.3.0]oct-1(5)-en-7-ium-3-ide",
        "CC(=O)[C-]1C[NH2+]CC1=O": "4-(acetyl)-3-oxo-pyrrolidin-1-ium-4-ide",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_von_baeyer_polycycle_keeps_descriptor_source_numbering():
    cases = {
        "C1C2C1C13COC21CO3": "7,9-dioxatetracyclo[3.2.2.0^{1,5}.0^{2,4}]nonane",
        "C1NC23COC12C=CC3": "9-oxa-7-azatricyclo[3.2.2.0^{1,5}]non-3-ene",
        "C1NC23COC12COC3": "3,9-dioxa-7-azatricyclo[3.2.2.0^{1,5}]nonane",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_substituted_alkoxy_prefixes_preserve_imino_ether_connectivity():
    assert name_smiles("OCCOC(=N)NC=O") == "N-((2-hydroxyethoxy)(imino)methyl)formamide"


def test_central_hydride_alkoxy_ligands_are_graph_derived():
    cases = {
        "COP(OC)OC": "trimethoxy-phosphane",
        "CCOP(OCC)OCC": "triethoxy-phosphane",
        "CC(C)OP(OC(C)C)OC(C)C": "triisopropoxy-phosphane",
        "CC(C)(C)O[PH](OC(C)(C)C)OC(C)(C)C": "tris(tert-butoxy)-lambda4-phosphane",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_rooted_alkyl_ligands_are_derived_as_systematic_substituents():
    cases = {
        "CC(C)OP": "propan-2-yl",
        "CC(C)(C)OP": "2-methylpropan-2-yl",
    }

    for smiles, expected in cases.items():
        mol = read_smiles(smiles)
        oxygen = next(idx for idx, atom in mol.atoms.items() if atom.symbol == "O")
        phosphorus = next(idx for idx, atom in mol.atoms.items() if atom.symbol == "P")
        root = next(idx for idx in mol.get_neighbors(oxygen) if idx != phosphorus and mol.atoms[idx].symbol == "C")
        assert _alkyl_ligand_name(mol, set(mol.atoms), root, oxygen) == expected


def test_trace_segment_schema_for_functionalized_molecules():
    for smiles in ["CCO", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]:
        analysis = analyze_smiles(smiles)
        assert analysis.trace_segments
        for segment in analysis.trace_segments:
            assert {"key", "label", "atoms", "bonds", "name_terms", "rule_hint"} <= set(segment)
            assert isinstance(segment["atoms"], list)
            assert isinstance(segment["bonds"], list)
            assert isinstance(segment["name_terms"], list)


def test_analysis_exposes_operation_ledger():
    acid = analyze_smiles("CC(=O)O")
    alkene = analyze_smiles("C=C")

    assert any(operation.operation_class == OperationClass.SUBSTITUTIVE for operation in acid.operations)
    assert any(operation.detail == "principal_group_and_substituent_assembly" for operation in acid.operations)
    assert any(operation.operation_class == OperationClass.SUBTRACTIVE for operation in alkene.operations)
