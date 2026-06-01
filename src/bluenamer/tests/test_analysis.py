import pytest

from bluenamer import (
    RULES,
    NamingEngine,
    NamingIntent,
    NamingRequest,
    OperationClass,
    TracePhase,
    analyze_smiles,
    name_smiles,
)
from bluenamer.additive import add_indicated_hydrogens
from bluenamer.assembler import (
    assemble_name,
    post_process_name,
)
from bluenamer.assembly_charge import parent_charge_operations
from bluenamer.assembly_parts import (
    AssemblyParts,
    NameAtomBinding,
    ParentChargeItem,
    SubstituentItem,
    UnsaturationItem,
)
from bluenamer.assembly_spiro import (
    _spiro_side_locant,
    extract_spiro_side_prefixes,
    format_spiro_core,
    split_spiro_substituents,
)
from bluenamer.charge_pair_roles import charge_pair_roles, unsupported_charge_pair_roles
from bluenamer.chains import find_all_carbon_paths, find_ring_systems
from bluenamer.formatting import ensure_stereo_descriptor_boundary, format_counted_prefixes
from bluenamer.fused_topology import (
    RingTopologyRoute,
    RingTopologyRouteKind,
    bridged_fused_candidates,
    charged_fused_template_gate,
    classify_ring_topology_route,
    fused_component_from_retained_match,
    fused_component_registry,
    fused_emission_examples,
    fused_numbering_from_retained_match,
    fused_parent_side_letters,
    spiro_component_reference,
)
from bluenamer.fused_ion_templates import fused_ion_operation_candidates, fused_ion_template_registry
from bluenamer.functional_groups import (
    PERCEPTION_DETECTORS,
    PERCEPTION_SPECS,
    PerceptionDetectorSpec,
    register_group_detector,
    register_perception_spec,
)
from bluenamer.graph_io import read_smiles
from bluenamer.heteroatom_subgraphs import name_heteroatom_subgraph
from bluenamer.heteroatom_substituent_specs import central_oxo_substituent_prefix, ligand_prefix, unsubstituted_prefix
from bluenamer.heterocumulene_roles import nitrogen_heterocumulene_role
from bluenamer.hypervalent_roles import HypervalentLigandRole, hypervalent_center_role, hypervalent_center_roles
from bluenamer.ionic_naming import (
    apply_parent_charge_names,
    apply_retained_parent_ide,
    apply_ring_parent_nitrogen_zwitterion_stack,
    apply_terminal_parent_ide,
    contains_invalid_locant_ide,
    parent_charge_sites,
)
from bluenamer.locants import as_display_locant, coerce_display_numbering, locant_text, parse_locant
from bluenamer.molecule import Molecule
from bluenamer.name_bindings import postprocess_name_atom_bindings
from bluenamer.name_operations import HydroOperation, ParentSuffixOperation
from bluenamer.name_postprocessing import (
    apply_connection_boundary_postprocessing,
    postprocessing_rule_inventory,
)
from bluenamer.namer import _number_saturated_n_ring_for_spiro, _spiro_subgraph_assembly, name_component, name_subgraph
from bluenamer.naming_audit import UnnamedAtomError, assert_component_fully_named, audit_charge_pair_templates
from bluenamer.naming_data import namer_rules
from bluenamer.nitrogen_roles import (
    acid_derived_hydrazone_roles,
    azine_roles,
    nitrogen_chain_roles,
    terminal_n3_substituent_role,
)
from bluenamer.oxoacid_roles import OxoLigandRole, central_oxo_roles, central_oxo_substituent_role
from bluenamer.oxoacid_templates import OxoacidTemplateKind, oxoacid_role_template
from bluenamer.opsin_resource_data import (
    oxoacid_ester_suffix_templates,
    opsin_resource_grammar,
    retained_fused_derivative_gate,
    retained_fused_token,
    retained_fused_token_status,
    retained_fused_tokens,
)
from bluenamer.numbering import NUMBERING_CRITERIA, NumberingPreference, polycycle_numbering_key
from bluenamer.parent_pipeline import build_parent_assembly_plan
from bluenamer.parent_selection import (
    PARENT_SELECTION_CRITERIA,
    ParentCandidate,
    ParentSelection,
    ParentSeniorityProfile,
    select_principal_parent,
)
from bluenamer.peroxy_carbonyl_roles import PeroxyCarbonylKind, peroxy_carbonyl_roles
from bluenamer.perception import PerceivedGroup, perceive_groups
from bluenamer.polycycle_topology import (
    audit_von_baeyer_descriptor,
    bicyclo_proof,
    build_ring_numbering,
    linear_dispiro_proof,
    monospiro_proof,
    ring_system_topology,
)
from bluenamer.principal_suffixes import render_principal_suffix
from bluenamer.resonance_compare import equivalent_smiles
from bluenamer.retained_specs import retained_parent_spec
from bluenamer.retained_fused_templates import (
    RetainedFusedGraphTemplate,
    match_retained_fused_template,
    match_retained_fused_templates,
    pending_retained_fused_parent_names,
    retained_fused_graph_templates,
    retained_fused_template_from_data,
    template_molecule,
)
from bluenamer.ring_parent import RingParent
from bluenamer.role_certificate import audit_role_certificate, certificate_from_perceived_group, certificates_from_assembly
from bluenamer.ring_renderer import render_ring_descriptor, render_von_baeyer_descriptor
from bluenamer.ring_systems import ring_system_fragment
from bluenamer.rule_layout import rule_group_specs, rule_groups, section_group_map, unassigned_sections
from bluenamer.rules.retained import get_retained_ring
import bluenamer.rules.retained as retained_rules
from bluenamer.special_cases import _alkyl_ligand_name, structural_replacement_parent_name
from bluenamer.spiro_assembly import SpiroAssembly
from bluenamer.stereo_audit import audit_stereochemistry
from bluenamer.suffix_stack import SuffixStack
from bluenamer.von_baeyer import _classify_secondary_bridges, find_von_baeyer_candidates


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


def test_repeated_substituent_with_internal_multiplier_uses_complex_multiplier():
    assert format_counted_prefixes(["dihydroxyphosphoryl", "dihydroxyphosphoryl"]) == "bis(dihydroxyphosphoryl)"


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
        ("P", 3, 1, 1, 0): "methyl dihydrogen phosphate",
        ("S", 2, 2, 2, 0): "ethyl hydrogen sulfate",
        ("N", 2, 1, 1, 1): "methyl nitrate",
        ("C", 2, 1, 2, 0): "ethyl hydrogen carbonate",
    }

    for (central_symbol, single_o, double_o, carbon_count, central_charge), expected in cases.items():
        mol = _oxoacid_ester_graph(central_symbol, single_o, double_o, carbon_count, central_charge)
        assert structural_replacement_parent_name(mol, set(mol.atoms)) == expected


def test_charge_normalized_halogen_oxoacid_esters_match_acid_ester_specs():
    acid = _halogen_oxoacid_graph("Br", single_o=1, charged_oxo_o=2, central_charge=2)
    ester = _halogen_oxoacid_ester_graph("Br", charged_oxo_o=2, carbon_count=1, central_charge=2)

    assert structural_replacement_parent_name(acid, set(acid.atoms)) == "bromic acid"
    assert structural_replacement_parent_name(ester, set(ester.atoms)) == "methyl bromate"


def test_oxoacid_ester_suffixes_are_role_template_backed():
    phosphate = _oxoacid_ester_graph("P", single_o=3, double_o=1, carbon_count=1)
    phosphate_role = central_oxo_roles(phosphate, set(phosphate.atoms))[0]

    template = oxoacid_role_template(phosphate, phosphate_role)

    assert template is not None
    assert template.key == "phosphate_monoester_neutral"
    assert template.kind == OxoacidTemplateKind.ESTER_SUFFIX
    assert template.opsin_verified
    assert template.preserves_formal_charges
    assert structural_replacement_parent_name(phosphate, set(phosphate.atoms)) == "methyl dihydrogen phosphate"


def test_charge_normalized_halogen_oxoacid_esters_are_template_classified():
    ester = _halogen_oxoacid_ester_graph("Cl", charged_oxo_o=2, carbon_count=1, central_charge=2)
    role = central_oxo_roles(ester, set(ester.atoms))[0]

    template = oxoacid_role_template(ester, role)

    assert template is not None
    assert template.key == "charge_normalized_halogen_oxoester"
    assert template.kind == OxoacidTemplateKind.ESTER_SUFFIX
    assert template.opsin_verified
    assert not template.preserves_formal_charges
    assert structural_replacement_parent_name(ester, set(ester.atoms)) == "methyl chlorate"


def test_charge_normalized_halogen_peroxy_roles_use_common_peroxyhalate_template():
    peroxy = _halogen_peroxy_oxo_graph("Cl", charged_oxo_o=2, central_charge=2)
    role = central_oxo_roles(peroxy, set(peroxy.atoms))[0]

    template = oxoacid_role_template(peroxy, role)

    assert template is not None
    assert template.key == "charge_normalized_chlorine_peroxy_oxoester"
    assert template.kind == OxoacidTemplateKind.ESTER_SUFFIX
    assert template.opsin_verified
    assert template.preserves_formal_charges
    assert structural_replacement_parent_name(peroxy, set(peroxy.atoms)) == "methyl peroxychlorate"


def test_mixed_central_hydride_ligands_are_boundary_protected():
    assert name_smiles("CCOSCl") == "(chloro)(ethoxy)sulfane"


def test_oxoacid_esters_use_recursive_front_modifier_namer():
    ester = _halogen_oxoacid_ester_graph("Br", charged_oxo_o=2, carbon_count=1, central_charge=2)
    calls = []

    def branch_namer(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None = None):
        calls.append((start_idx, exclude_atoms, upstream_atom))
        return "(custom modifier)"

    assert structural_replacement_parent_name(ester, set(ester.atoms), branch_namer) == "custom modifier bromate"
    assert calls == [(2, {0, 1, 3, 4}, 1)]


def test_central_oxo_role_classifies_acid_ligands_from_graph():
    mol = _oxoacid_graph("P", single_o=3, double_o=1)

    roles = central_oxo_roles(mol, set(mol.atoms))

    assert len(roles) == 1
    assert roles[0].central_symbol == "P"
    assert roles[0].count(OxoLigandRole.HYDROXY) == 3
    assert roles[0].count(OxoLigandRole.OXO) == 1
    assert roles[0].spec_counts() == (3, 1)


def test_central_oxo_role_classifies_alkoxy_and_oxido_ligands_from_graph():
    mol = _halogen_oxoacid_ester_graph("Br", charged_oxo_o=2, carbon_count=1, central_charge=2)

    roles = central_oxo_roles(mol, set(mol.atoms))

    assert len(roles) == 1
    assert roles[0].count(OxoLigandRole.ALKOXY) == 1
    assert roles[0].count(OxoLigandRole.OXO) == 2
    assert roles[0].spec_counts() == (1, 2)


def test_central_oxo_parent_names_full_anions_from_role_data():
    mol = Molecule()
    mol.add_atom("S", 0)
    mol.add_atom("O", 1, charge=-1)
    mol.add_atom("O", 2, charge=-1)
    mol.add_atom("O", 3)
    mol.add_atom("O", 4)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(0, 2, order=1)
    mol.add_bond(0, 3, order=2)
    mol.add_bond(0, 4, order=2)

    assert structural_replacement_parent_name(mol, set(mol.atoms)) == "sulfate"


def test_neutral_oxoacid_ester_suffix_preserves_remaining_hydrogens():
    phosphate = _oxoacid_ester_graph("P", single_o=3, double_o=1, carbon_count=1)
    sulfate = _oxoacid_ester_graph("S", single_o=2, double_o=2, carbon_count=2)
    carbonate = _oxoacid_ester_graph("C", single_o=2, double_o=1, carbon_count=3)

    assert structural_replacement_parent_name(phosphate, set(phosphate.atoms)) == "methyl dihydrogen phosphate"
    assert structural_replacement_parent_name(sulfate, set(sulfate.atoms)) == "ethyl hydrogen sulfate"
    assert structural_replacement_parent_name(carbonate, set(carbonate.atoms)) == "propyl hydrogen carbonate"


def test_central_oxo_substituent_role_allows_non_oxygen_ligands():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("S", 1)
    mol.add_atom("O", 2)
    mol.add_atom("O", 3, charge=-1)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=2)
    mol.add_bond(1, 3, order=1)

    role = central_oxo_substituent_role(mol, set(mol.atoms), 1)

    assert role is not None
    assert role.central_symbol == "S"
    assert role.count(OxoLigandRole.OXO) == 1
    assert role.count(OxoLigandRole.OXIDO) == 1


def test_central_oxo_substituent_class_rendering_is_data_backed():
    mol = _phosphorus_oxo_substituent_graph(oxo_count=2)
    role = central_oxo_substituent_role(mol, set(mol.atoms), 1)

    assert role is not None
    assert central_oxo_substituent_prefix(role) == "dioxophosphanyl"
    assert name_heteroatom_subgraph(mol, 1, exclude_atoms={0}, upstream_atom=0, branch_namer=_empty_branch_namer) == (
        "dioxophosphanyl"
    )


def test_central_oxo_substituent_class_falls_back_when_unconfigured():
    mol = _phosphorus_oxo_substituent_graph(oxo_count=3)
    role = central_oxo_substituent_role(mol, set(mol.atoms), 1)

    assert role is not None
    assert central_oxo_substituent_prefix(role) is None
    assert name_heteroatom_subgraph(mol, 1, exclude_atoms={0}, upstream_atom=0, branch_namer=_empty_branch_namer) == (
        "phosphoryl"
    )


def test_central_oxo_substituent_class_renders_one_oxo_phosphorus_explicitly():
    mol = _phosphorus_oxo_substituent_graph(oxo_count=1)
    role = central_oxo_substituent_role(mol, set(mol.atoms), 1)

    assert role is not None
    assert central_oxo_substituent_prefix(role) == "oxophosphanyl"
    assert name_heteroatom_subgraph(mol, 1, exclude_atoms={0}, upstream_atom=0, branch_namer=_empty_branch_namer) == (
        "oxophosphanyl"
    )


def test_central_oxo_substituent_class_handles_sulfo_signature():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("S", 1)
    mol.add_atom("O", 2)
    mol.add_atom("O", 3)
    mol.add_atom("O", 4)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=2)
    mol.add_bond(1, 3, order=2)
    mol.add_bond(1, 4, order=1)

    role = central_oxo_substituent_role(mol, set(mol.atoms), 1)

    assert role is not None
    assert central_oxo_substituent_prefix(role) == "sulfo"
    assert name_heteroatom_subgraph(mol, 1, exclude_atoms={0}, upstream_atom=0, branch_namer=_empty_branch_namer) == "sulfo"


def test_central_oxo_substituent_class_excludes_implied_hydroxy_ligands():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("P", 1)
    mol.add_atom("O", 2)
    mol.add_atom("O", 3)
    mol.add_atom("O", 4)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=2)
    mol.add_bond(1, 3, order=1)
    mol.add_bond(1, 4, order=1)

    role = central_oxo_substituent_role(mol, set(mol.atoms), 1)

    assert role is not None
    assert central_oxo_substituent_prefix(role) == "dihydroxyphosphoryl"
    assert name_heteroatom_subgraph(mol, 1, exclude_atoms={0}, upstream_atom=0, branch_namer=_empty_branch_namer) == (
        "dihydroxyphosphoryl"
    )


def test_nitrogen_heterocumulene_role_precedes_formamido_fallback():
    mol = Molecule()
    mol.add_atom("P", 0)
    mol.add_atom("N", 1)
    mol.add_atom("C", 2)
    mol.add_atom("O", 3)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=2)
    mol.add_bond(2, 3, order=2)

    role = nitrogen_heterocumulene_role(mol, 1, exclude_atoms={0}, upstream_atom=0)

    assert role is not None
    assert role.prefix == "isocyanato"
    assert role.atom_ids == {1, 2, 3}
    assert name_heteroatom_subgraph(mol, 1, exclude_atoms={0}, upstream_atom=0, branch_namer=_empty_branch_namer) == (
        "isocyanato"
    )


def _phosphorus_oxo_substituent_graph(oxo_count: int) -> Molecule:
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("P", 1)
    mol.add_bond(0, 1, order=1)
    for offset in range(oxo_count):
        oxygen = offset + 2
        mol.add_atom("O", oxygen)
        mol.add_bond(1, oxygen, order=2)
    return mol


def _empty_branch_namer(_mol: Molecule, _start_idx: int, _exclude_atoms: set[int], _upstream_atom: int | None = None):
    return ""


def test_nitrogen_chain_roles_classify_azido_from_graph():
    mol = _carbon_bound_n3_graph(attachment_order=1, first_nn_order=2, second_nn_order=2)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("azido", 0, {1, 2, 3})
    ]
    role = roles[0]
    assert role.ordered_atoms == (0, 1, 2, 3)
    assert role.bond_orders == (1, 2, 2)
    assert role.charge_pattern == (0, 0, 0, 0)


def test_terminal_thioaldehyde_on_ring_uses_carbothialdehyde_suffix():
    generated = name_smiles("Cc1cc(Cl)c(C=S)c(Cl)n1")

    assert generated == "2,4-dichloro-6-methylpyridine-3-carbothialdehyde"


def test_terminal_thioaldehyde_prefix_uses_monovalent_thioformyl():
    generated = name_smiles("NC(=O)c1cccc(N)c1C=S")

    assert generated == "3-amino-2-thioformylbenzene-1-carboxamide"


def test_terminal_thioformyl_subgraph_is_graph_derived():
    mol = Molecule()
    mol.add_atom("N", 0)
    mol.add_atom("C", 1)
    mol.add_atom("S", 2)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=2)

    assert name_subgraph(mol, 1, {0}, upstream_atom=0) == "thioformyl"


def test_thioester_is_not_perceived_as_ring_thioaldehyde():
    generated = name_smiles("CCSC(=S)C1=CCC=CN1")

    assert generated == "2-((ethylsulfanyl)(thioxo)methyl)-1-azacyclohexa-2,5-diene"


def test_cyclic_thioamide_is_not_perceived_as_thioaldehyde():
    generated = name_smiles("CCCCCCCCCCCCCC(=S)N1CCCCC1")

    assert generated == "1-(1-thioxotetradecyl)piperidine"


def test_cyclic_sulfone_ring_is_not_perceived_as_sulfonate_suffix():
    generated = name_smiles("C=CC(=O)NC1C(C)COS1(=O)=O")

    assert generated == "N-(4-methyl-2,2-dioxo-1-oxa-2lambda^6-thiacyclopentan-3-yl)prop-2-enamide"


def test_nitrile_oxide_is_not_rendered_as_isocyano():
    generated = name_smiles("Nc1ccccc1C#[N+][O-]")

    assert generated == "2-aminobenzene-1-carbonitrile oxide"


def test_direct_phosphonic_acid_substituent_keeps_p_c_bond():
    generated = name_smiles("NCC(O)C(O)P(=O)(O)O")

    assert generated == "3-amino-1-(dihydroxyphosphoryl)propane-1,2-diol"


def test_terminal_carbon_phosphorus_triple_substituent_keeps_triple_bond():
    generated = name_smiles("CC(=O)NC#P")

    assert generated == "N-phosphanylidynemethylacetamide"


def test_terminal_oxophosphorus_triple_substituent_keeps_triple_bond():
    generated = name_smiles("O=P#Cc1ccncc1")

    assert generated == "4-oxophosphanylidynemethylpyridine"


def test_oxophosphorus_triple_substituent_through_sulfur_keeps_triple_bond():
    generated = name_smiles("NCC(S)S#P=O")

    assert generated == "2-amino-1-(oxophosphanylidynesulfanyl)ethane-1-thiol"


def test_upstream_sulfur_imide_attachment_renders_sulfinyl_not_sulfonimidoylidene():
    generated = name_smiles("NC(=S)N=S=O")

    assert generated == "1-amino-N-sulfinylidenemethanethioamide"


def test_oxosulfur_imide_branch_preserves_n_s_double_bond():
    mol = Molecule()
    for idx, symbol in enumerate(["C", "N", "S", "O", "O"]):
        mol.add_atom(symbol, idx)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=2)
    mol.add_bond(2, 3, order=2)
    mol.add_bond(2, 4, order=2)

    def empty_branch_namer(*args, **kwargs):
        return ""

    assert name_heteroatom_subgraph(mol, 1, {0}, 0, empty_branch_namer) == "((dioxosulfanylidene)amino)"


def test_chiral_sulfoxide_parent_keeps_sulfur_descriptor():
    assert name_smiles("CC[S@@](=O)CC(C)C") == "(R)-2-methylpropyl ethyl sulfoxide"
    assert name_smiles("CCC[S@@](=O)CC") == "(S)-ethyl propyl sulfoxide"


def test_repeated_cyanoalkyl_front_modifiers_use_complex_multiplier():
    generated = name_smiles("CC(C(=O)OCC#N)C(=O)OCC#N")

    assert generated == "bis(2-nitriloethyl) 2-methylpropanedioate"


def test_simple_azine_uses_hydrazone_template_not_hydrazinyl_prefix():
    generated = name_smiles(r"CC/C(C)=N\N=C(C)C")

    assert generated == "(2Z)-butan-2-one propan-2-ylidenehydrazone"


def test_azine_uses_recursive_ylidene_side_for_complex_branch():
    generated = name_smiles(r"CC(C)=N/N=C(\C)C(C)c1cnccn1")

    assert generated == "propan-2-one (2E)-3-(pyrazin-2-yl)butan-2-ylidenehydrazone"


def test_azine_roles_preserve_ordered_sides_from_graph():
    mol = Molecule()
    for idx, symbol in enumerate(["C", "N", "N", "C", "C", "C"]):
        mol.add_atom(symbol, idx)
    mol.add_bond(0, 1, order=2)
    mol.add_bond(1, 2, order=1)
    mol.add_bond(2, 3, order=2)
    mol.add_bond(0, 4, order=1)
    mol.add_bond(3, 5, order=1)

    roles = azine_roles(mol, set(range(6)))

    assert len(roles) == 1
    role = roles[0]
    assert role.ordered_atoms == (0, 1, 2, 3)
    assert role.left.side_atoms == frozenset({0, 4})
    assert role.right.side_atoms == frozenset({3, 5})
    assert [(segment.start_atom, segment.end_atom, segment.bond_order) for segment in role.segments] == [
        (0, 1, 2),
        (1, 2, 1),
        (2, 3, 2),
    ]


def test_hydrazone_prefix_with_terminal_carbon_imino_keeps_connectivity():
    generated = name_smiles(r"COC(=O)CC(C)=N/N=C(\N)SC")

    assert generated == "methyl 3-((((1E)-(amino)(methylsulfanyl)methylidene)amino)imino)butanoate"


def test_hydrazone_role_preserves_parent_side_metadata():
    mol = _carbon_bound_n3_graph(attachment_order=2, first_nn_order=1, second_nn_order=2)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())
    role = roles[0]

    assert role.key == "aldehyde_hydrazone"
    assert role.ordered_atoms == (0, 1, 2)
    assert role.bond_orders == (2, 1)
    assert role.hydrazone_side is not None
    assert role.hydrazone_side.carbon_atom == 0
    assert role.hydrazone_side.nitrogen_atom == 1
    assert role.hydrazone_side.attachment_atom == 0
    assert role.hydrazone_side.parent_kind == "aldehyde_hydrazone"


def test_amidinohydrazone_tail_is_part_of_principal_group():
    generated = name_smiles(r"Cc1cccc(Cl)c1/C=N\N=C(N)N")

    assert generated == "2-chloro-6-methylbenzene-1-carbaldehyde amidinohydrazone"


def test_hydrazone_suffix_preserves_unlocanted_ez_stereo():
    generated = name_smiles(r"N/N=C/c1ccc(N)cc1S(N)(=O)=O")

    assert generated == "(E)-4-amino-2-(sulfamoyl)benzaldehyde hydrazone"


def test_hydrazone_stereo_does_not_duplicate_unlocanted_descriptor():
    generated = name_smiles(r"C/C=N\Nc1nc2ccccc2o1")

    assert generated == "(Z)-N-(benzoxazol-2-yl)acetaldehyde hydrazone"


def test_hydrazone_n_substituent_uses_suffix_modifier_for_nitrogen_parent():
    generated = name_smiles("c1ccc(NN=C2CCNCC2)cc1")

    assert generated == "piperidin-4-one phenylhydrazone"


def test_nitrogen_chain_roles_classify_neutral_diazene_amino_prefix():
    mol = _carbon_bound_n3_graph(attachment_order=1, first_nn_order=2, second_nn_order=1)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("aminodiazenyl", 0, {1, 2, 3})
    ]


def test_nitrogen_chain_roles_classify_neutral_diazenylamino_prefix():
    mol = _carbon_bound_n3_graph(attachment_order=1, first_nn_order=1, second_nn_order=2)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("diazenylamino", 0, {1, 2, 3})
    ]


def test_nitrogen_chain_roles_do_not_call_neutral_triazane_azido():
    mol = _carbon_bound_n3_graph(attachment_order=1, first_nn_order=1, second_nn_order=1)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("hydrazinylamino", 0, {1, 2, 3})
    ]


def test_nitrogen_chain_roles_keep_imino_hydrazone_distinct_from_azido():
    mol = _carbon_bound_n3_graph(attachment_order=2, first_nn_order=1, second_nn_order=2)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert not any(role.key == "azido" for role in roles)
    assert [(role.key, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("aldehyde_hydrazone", 0, {1, 2})
    ]


def test_nitrogen_chain_roles_split_acid_derived_hydrazonamide_from_hydrazone():
    mol = Molecule()
    for idx, symbol in enumerate(["C", "N", "N", "N", "N"]):
        mol.add_atom(symbol, idx)
    mol.add_bond(0, 1, order=2)
    mol.add_bond(0, 2, order=1)
    mol.add_bond(0, 3, order=1)
    mol.add_bond(3, 4, order=1)

    roles = acid_derived_hydrazone_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.attachment_atom, set(role.atom_ids), role.bond_orders) for role in roles] == [
        ("hydrazonamide", 0, {0, 1, 2, 3, 4}, (2, 1, 1))
    ]
    assert not any(role.key.endswith("hydrazone") for role in roles)


def test_nitrogen_chain_roles_split_imidohydrazide_and_thiohydrazide_from_hydrazone():
    imido = Molecule()
    for idx, symbol in enumerate(["C", "N", "N", "N"]):
        imido.add_atom(symbol, idx)
    imido.add_bond(0, 1, order=2)
    imido.add_bond(0, 2, order=1)
    imido.add_bond(2, 3, order=1)

    thio = Molecule()
    for idx, symbol in enumerate(["C", "S", "N", "N"]):
        thio.add_atom(symbol, idx)
    thio.add_bond(0, 1, order=2)
    thio.add_bond(0, 2, order=1)
    thio.add_bond(2, 3, order=1)

    imido_roles = acid_derived_hydrazone_roles(imido, cyclic_atoms=set())
    thio_roles = acid_derived_hydrazone_roles(thio, cyclic_atoms=set())

    assert [(role.key, set(role.atom_ids), role.bond_orders) for role in imido_roles] == [
        ("imidohydrazide", {0, 1, 2, 3}, (2, 1, 1))
    ]
    assert [(role.key, set(role.atom_ids), role.bond_orders) for role in thio_roles] == [
        ("thiohydrazide", {0, 1, 2, 3}, (2, 1, 1))
    ]


def test_nitrogen_chain_roles_classify_charged_diazonio_from_graph():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("N", 1, charge=1)
    mol.add_atom("N", 2, charge=-1)
    mol.add_bond(0, 1, order=2)
    mol.add_bond(1, 2, order=2)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("diazo", 0, {1, 2})
    ]
    assert roles[0].charge_pattern == (0, 1, -1)
    assert roles[0].bond_orders == (2, 2)


def test_nitrogen_chain_roles_classify_singly_attached_charged_n2_as_diazonio():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("N", 1, charge=1)
    mol.add_atom("N", 2)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=3)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("diazonio", 0, {1, 2})
    ]


def test_neutral_single_bonded_diazene_prefix_is_not_diazo():
    generated = name_smiles("CCOC(=O)C(C#N)N=N")

    assert generated == "ethyl 2-cyano-2-(diazenyl)acetate"


def test_nitrogen_chain_roles_allow_terminal_imino_hydrazone():
    mol = _carbon_bound_n3_graph(attachment_order=2, first_nn_order=1, second_nn_order=2)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert any(role.key.endswith("hydrazone") for role in roles)


def test_nitrogen_chain_roles_allow_n_substituted_hydrazones():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("N", 1)
    mol.add_atom("N", 2)
    mol.add_atom("C", 3)
    mol.add_atom("C", 4)
    mol.add_bond(0, 1, order=2)
    mol.add_bond(1, 2, order=1)
    mol.add_bond(2, 3, order=1)
    mol.add_bond(2, 4, order=1)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("aldehyde_hydrazone", 0, {1, 2})
    ]


def test_nitrogen_chain_roles_classify_hydrazine_from_graph():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("N", 1)
    mol.add_atom("N", 2)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=1)

    roles = nitrogen_chain_roles(mol, cyclic_atoms=set())

    assert [(role.key, role.is_principal_candidate, role.attachment_atom, set(role.atom_ids)) for role in roles] == [
        ("hydrazine", False, 0, {1, 2})
    ]


def test_nitrogen_chain_roles_make_cyclic_hydrazines_prefixes():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("N", 1)
    mol.add_atom("N", 2)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=1)

    roles = nitrogen_chain_roles(mol, cyclic_atoms={0})

    assert [(role.key, role.is_principal_candidate, role.variant) for role in roles] == [
        ("hydrazine", False, "prefix")
    ]


def test_cyclic_hydrazines_render_as_hydrazinyl_prefixes():
    generated = name_smiles("NNc1cn[nH]c1")

    assert generated == "4-hydrazinyl-1H-pyrazole"


def test_terminal_n3_substituent_role_preserves_charge_and_bond_pattern():
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("N", 1)
    mol.add_atom("N", 2, charge=1)
    mol.add_atom("N", 3, charge=-1)
    mol.add_bond(0, 1, order=1)
    mol.add_bond(1, 2, order=2)
    mol.add_bond(2, 3, order=2)

    role = terminal_n3_substituent_role(mol, 1, {0}, 0)

    assert role is not None
    assert role.key == "azido"
    assert role.ordered_atoms == (0, 1, 2, 3)
    assert role.charge_pattern == (0, 0, 1, -1)
    assert role.bond_orders == (1, 2, 2)


def test_substituted_cyclic_hydrazines_keep_n_ligands_in_prefix():
    generated = name_smiles("CN(C)Nc1cccc2ncccc12")

    assert generated == "5-(N',N'-dimethylhydrazinyl)quinoline"


def test_perception_consumes_nitrogen_chain_roles_before_legacy_azido():
    mol = _carbon_bound_n3_graph(attachment_order=2, first_nn_order=1, second_nn_order=2)

    groups = perceive_groups(mol)

    assert not any(group.key == "azido" for group in groups)


def _carbon_bound_n3_graph(attachment_order: int, first_nn_order: int, second_nn_order: int) -> Molecule:
    mol = Molecule()
    mol.add_atom("C", 0)
    mol.add_atom("N", 1)
    mol.add_atom("N", 2)
    mol.add_atom("N", 3)
    mol.add_bond(0, 1, order=attachment_order)
    mol.add_bond(1, 2, order=first_nn_order)
    mol.add_bond(2, 3, order=second_nn_order)
    return mol


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


def _halogen_peroxy_oxo_graph(central_symbol: str, charged_oxo_o: int, central_charge: int) -> Molecule:
    mol = Molecule()
    mol.add_atom(symbol=central_symbol, idx=0, charge=central_charge)
    mol.add_atom(symbol="O", idx=1)
    mol.add_bond(0, 1, order=1)
    mol.add_atom(symbol="O", idx=2)
    mol.add_bond(1, 2, order=1)
    mol.add_atom(symbol="C", idx=3)
    mol.add_bond(2, 3, order=1)
    next_idx = 4
    for _ in range(charged_oxo_o):
        mol.add_atom(symbol="O", idx=next_idx, charge=-1)
        mol.add_bond(0, next_idx, order=1)
        next_idx += 1
    return mol


def test_ring_descriptor_rendering_is_registry_backed():
    assert render_ring_descriptor("spiro", (2, 3)) == "spiro[2.3]"
    assert render_ring_descriptor("bicyclo", (2, 1, 0)) == "bicyclo[2.1.0]"
    assert render_von_baeyer_descriptor(2, "[3.2.2.0^{1,5}]") == "tricyclo[3.2.2.0^{1,5}]"
    assert render_von_baeyer_descriptor(6, "[2.2.1.0^{2,6}.0^{2,7}.0^{3,5}.0^{3,7}.0^{5,7}]").startswith(
        "heptacyclo["
    )
    assert render_von_baeyer_descriptor(10, "[2.2.1.0^{2,6}]").startswith("undecacyclo[")
    assert render_von_baeyer_descriptor(20, "[2.2.1.0^{2,6}]").startswith("henicosacyclo[")


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


def test_role_certificate_projects_perceived_group_graph_metadata():
    mol = read_smiles("CC(=O)[O-]")
    groups = perceive_groups(mol)

    carboxylate = next(group for group in groups if group.key == "carboxylate")
    certificate = certificate_from_perceived_group(mol, carboxylate)

    assert certificate.key == "carboxylate"
    assert certificate.represented_atoms == carboxylate.atom_ids
    assert certificate.represented_bonds == carboxylate.bond_ids
    assert certificate.represented_charges == {3: -1}
    assert certificate.missing_charged_atoms(mol, carboxylate.atom_ids) == set()


def test_role_certificate_projects_assembled_name_bindings():
    mol = read_smiles("[NH4+]")
    parts = AssemblyParts(
        parent_length=1,
        parent_atom_ids={0},
        parent_charges=[ParentChargeItem(locant="1", symbol="N", charge=1, atom_id=0)],
    )

    certificates = certificates_from_assembly(mol, parts)
    charge = next(certificate for certificate in certificates if certificate.key == "charge:parent_charge")

    assert charge.represented_atoms == {0}
    assert charge.represented_charges == {0: 1}
    assert charge.locants_by_atom == {0: "1"}


def test_charge_pair_roles_classify_supported_and_unsupported_templates():
    supported = read_smiles("C[S+](C)[CH-]C")
    supported_roles = charge_pair_roles(supported)

    sulfonium = next(role for role in supported_roles if role.key == "sulfonium_ylide_single_bond")
    assert sulfonium.template_supported
    assert sulfonium.certificate(supported).represented_charges == {1: 1, 3: -1}
    assert sulfonium.template_audit(supported).ok

    unsupported = read_smiles("C[SH+](C)=[C-]c1ccccc1")
    unsupported_roles = unsupported_charge_pair_roles(unsupported)

    assert [(role.key, role.template_supported) for role in unsupported_roles] == [
        ("sulfur_carbanion_resonance_charge_pair", False)
    ]


def test_charge_pair_roles_classify_n_oxide_and_diazonium_azanide():
    n_oxide = read_smiles("C[N+]([O-])C")
    n_oxide_role = next(role for role in charge_pair_roles(n_oxide) if role.key == "n_oxide")
    assert n_oxide_role.template_supported
    assert n_oxide_role.certificate(n_oxide).represented_charges == {1: 1, 2: -1}
    assert n_oxide_role.template_audit(n_oxide).ok

    diazonium = read_smiles("C[N+]=[N-]")
    diazonium_role = next(role for role in charge_pair_roles(diazonium) if role.key == "diazonium_azanide")
    assert diazonium_role.template_supported
    assert diazonium_role.atom_ids == {1, 2}
    assert diazonium_role.template_audit(diazonium).ok


def test_charge_pair_template_audit_blocks_unsupported_resonance_charge_pair():
    mol = read_smiles("C[SH+](C)=[C-]c1ccccc1")

    audit = audit_charge_pair_templates(mol, set(mol.atoms))

    assert not audit.ok
    assert [role.key for role in audit.unsupported_roles] == ["sulfur_carbanion_resonance_charge_pair"]


def test_phosphane_borane_zwitterion_is_classified_and_opsin_enabled():
    mol = read_smiles("[BH3-][P+](C)(C)C")
    role = next(role for role in charge_pair_roles(mol) if role.key == "phosphane_borane_zwitterion")

    assert role.template_supported
    assert role.template_audit(mol).ok


def test_phosphane_borane_zwitterions_render_connected_boranuide_parent():
    assert name_smiles("[BH3-][P+](C)(C)C") == "(trimethylphosphaniumyl)boranuide"
    assert name_smiles("[BH3-][P+](CC)(CC)CO") == "(diethyl(hydroxymethyl)phosphaniumyl)boranuide"


def test_hypervalent_center_role_classifies_oxo_hydroxy_and_charge_pair_ligands():
    mol = _halogen_oxoacid_graph("Cl", single_o=0, charged_oxo_o=2, central_charge=2)

    role = hypervalent_center_role(mol, set(mol.atoms), 0)

    assert role is not None
    assert role.center_symbol == "Cl"
    assert role.count(HypervalentLigandRole.OXO) == 2
    assert role.certificate(mol).represented_charges == {0: 2, 1: -1, 2: -1}
    assert role.template_audit(mol).ok


def test_hypervalent_center_role_classifies_imino_thioxo_peroxy_ligands():
    mol = Molecule()
    mol.add_atom("S", 0)
    mol.add_atom("N", 1)
    mol.add_atom("S", 2)
    mol.add_atom("O", 3)
    mol.add_atom("O", 4)
    mol.add_bond(0, 1, order=2)
    mol.add_bond(0, 2, order=2)
    mol.add_bond(0, 3, order=1)
    mol.add_bond(3, 4, order=1)

    role = hypervalent_center_roles(mol, set(mol.atoms))[0]

    assert role.count(HypervalentLigandRole.IMINO) == 1
    assert role.count(HypervalentLigandRole.THIOXO) == 1
    assert role.count(HypervalentLigandRole.PEROXY) == 1
    assert role.template_audit(mol).ok


def test_peroxy_carbonyl_roles_classify_hydroperoxide_peroxoate_and_carbonate_like():
    hydroperoxide = read_smiles("CC(=O)OO")
    hydro_role = peroxy_carbonyl_roles(hydroperoxide, set(hydroperoxide.atoms))[0]
    assert hydro_role.kind == PeroxyCarbonylKind.HYDROPEROXIDE
    assert hydro_role.certificate(hydroperoxide).represented_atoms == hydro_role.atom_ids
    assert hydro_role.template_audit(hydroperoxide).ok

    peroxoate = read_smiles("CC(=O)O[O-]")
    peroxoate_role = peroxy_carbonyl_roles(peroxoate, set(peroxoate.atoms))[0]
    assert peroxoate_role.kind == PeroxyCarbonylKind.PEROXOATE
    assert peroxoate_role.certificate(peroxoate).represented_charges == {4: -1}
    assert peroxoate_role.template_audit(peroxoate).ok

    carbonate = read_smiles("COC(=O)OOC")
    carbonate_role = peroxy_carbonyl_roles(carbonate, set(carbonate.atoms))[0]
    assert carbonate_role.kind == PeroxyCarbonylKind.CARBONATE_LIKE
    assert carbonate_role.template_audit(carbonate).ok


def test_terminal_peroxy_carbonyl_charge_selects_peroxoate_suffix():
    assert name_smiles("CC(=O)OO") == "ethaneperoxoic acid"
    assert name_smiles("CC(=O)O[O-]") == "ethaneperoxoate"


def test_role_certificate_audit_reports_missing_template_coverage():
    mol = read_smiles("C[N+]([O-])C")
    role = next(role for role in charge_pair_roles(mol) if role.key == "n_oxide")

    audit = audit_role_certificate(mol, role.certificate(mol), expected_atoms={1, 2, 3})

    assert not audit.ok
    assert "missing=[3]" in audit.audit_errors[0]


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
    from bluenamer.rules import substituents, suffixes

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


def test_saturated_n_ring_spiro_numbering_prefers_side_features_before_spiro_locant():
    mol = Molecule()
    for idx, symbol in enumerate(["N", "C", "C", "C", "C", "C", "C"]):
        mol.add_atom(symbol, idx=idx)
    for idx, (first, second) in enumerate([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 6)], start=1):
        mol.add_bond(first, second, order=1, idx=idx)

    locants = _number_saturated_n_ring_for_spiro(mol, [0, 1, 2, 3, 4, 5], n_atom=0, spiro_atom=2, sub_comp=set(range(7)))

    assert locants[1] == "2"
    assert locants[2] == "3"


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


def test_stage3_retained_fused_pin_forms_and_indicated_hydrogens_are_validated():
    cases = {
        "c1ccc2occc2c1": "1-benzofuran",
        "c1ccc2sccc2c1": "1-benzothiophene",
        "c1ccc2[nH]cnc2c1": "1H-benzimidazole",
        "c1ccc2[nH]ncc2c1": "1H-indazole",
        "c1ccc2[nH]ccc2c1": "1H-indole",
        "c1ccc2cc3ccccc3cc2c1": "anthracene",
        "c1ccc2c(c1)ccc1ccccc12": "phenanthrene",
        "c1cc2ccc3cccc4ccc(c1)c2c34": "pyrene",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def _naphthalene_graph_template_row() -> dict:
    locants = ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a")
    return {
        "name": "naphthalene",
        "pin": True,
        "priority": 1,
        "aliases": [],
        "fusion_prefix": "naphtho",
        "derivative_stem": "naphthalen",
        "template": {
            "enabled": False,
            "locants": locants,
            "atoms": [
                {"locant": locant, "symbol": "C", "aromatic": True, "fusion": locant in {"4a", "8a"}}
                for locant in locants
            ],
            "bonds": [
                {"locants": ["1", "2"], "bond_class": "aromatic"},
                {"locants": ["2", "3"], "bond_class": "aromatic"},
                {"locants": ["3", "4"], "bond_class": "aromatic"},
                {"locants": ["4", "4a"], "bond_class": "aromatic"},
                {"locants": ["4a", "8a"], "bond_class": "fusion"},
                {"locants": ["8a", "1"], "bond_class": "aromatic"},
                {"locants": ["4a", "5"], "bond_class": "aromatic"},
                {"locants": ["5", "6"], "bond_class": "aromatic"},
                {"locants": ["6", "7"], "bond_class": "aromatic"},
                {"locants": ["7", "8"], "bond_class": "aromatic"},
                {"locants": ["8", "8a"], "bond_class": "aromatic"},
            ],
            "rings": [
                ["1", "2", "3", "4", "4a", "8a"],
                ["4a", "5", "6", "7", "8", "8a"],
            ],
            "fusion_atoms": ["4a", "8a"],
            "peripheral_atoms": locants,
            "interior_atoms": [],
            "numbering_policy": "retained_template",
        },
    }


def test_retained_fused_graph_template_schema_validates_locant_graphs():
    template = retained_fused_template_from_data(_naphthalene_graph_template_row())

    assert isinstance(template, RetainedFusedGraphTemplate)
    assert template.name == "naphthalene"
    assert template.locants == ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a")
    assert template.fusion_atoms == ("4a", "8a")
    assert template.attached_prefix == "naphtho"
    assert not template.enabled


def test_retained_fused_graph_template_builds_local_graph():
    template = retained_fused_template_from_data(_naphthalene_graph_template_row())
    mol = template_molecule(template)

    assert len(mol.atoms) == 10
    assert len(mol.bonds) == 11
    assert all(atom.symbol == "C" for atom in mol.atoms.values())
    assert sum(1 for atom in mol.atoms.values() if atom.is_aromatic) == 10


def test_retained_fused_graph_template_match_returns_locant_map():
    template = retained_fused_template_from_data(_naphthalene_graph_template_row())
    mol = template_molecule(template)

    match = match_retained_fused_template(mol, set(mol.atoms), template)

    assert match is not None
    assert match.template.name == "naphthalene"
    assert match.locant_to_atom["1"] == 0
    assert match.atom_to_locant[4] == "4a"
    assert match.locant_to_atom["8a"] == 9
    assert match.matched_atoms == frozenset(mol.atoms)


def test_retained_fused_graph_template_data_file_validates_guarded_core_entries():
    templates = retained_fused_graph_templates(include_disabled=True)
    by_name = {template.name: template for template in templates}

    assert {
        "naphthalene",
        "quinoline",
        "isoquinoline",
        "1,5-naphthyridine",
        "1,6-naphthyridine",
        "1,7-naphthyridine",
        "1,8-naphthyridine",
        "2,6-naphthyridine",
        "2,7-naphthyridine",
        "quinazoline",
        "quinoxaline",
        "cinnoline",
        "phthalazine",
        "azulene",
        "1H-phenalene",
        "acenaphthylene",
        "fluoranthene",
        "1H-perimidine",
        "pteridine",
        "2H-isoindole",
        "indolizine",
        "1H-indole",
    } <= set(by_name)
    enabled_names = {template.name for template in retained_fused_graph_templates()}
    assert not (
        {
            "naphthalene",
            "quinoline",
            "isoquinoline",
            "1,5-naphthyridine",
            "1,6-naphthyridine",
            "1,7-naphthyridine",
            "1,8-naphthyridine",
            "2,6-naphthyridine",
            "2,7-naphthyridine",
            "quinazoline",
            "quinoxaline",
            "cinnoline",
            "phthalazine",
            "azulene",
            "1H-phenalene",
            "acenaphthylene",
            "fluoranthene",
            "1H-perimidine",
            "pteridine",
            "2H-isoindole",
            "indolizine",
            "1H-indole",
        }
        & enabled_names
    )
    production_derivative_names = {
        template.name
        for template in templates
        if template.derivative_production_enabled
    }
    assert production_derivative_names == {
        "naphthalene",
        "quinoline",
        "isoquinoline",
        "1,5-naphthyridine",
        "1,6-naphthyridine",
        "1,7-naphthyridine",
        "1,8-naphthyridine",
        "2,6-naphthyridine",
        "2,7-naphthyridine",
        "quinazoline",
        "quinoxaline",
        "cinnoline",
        "phthalazine",
    }
    assert by_name["1H-indole"].default_indicated_h == ("1",)
    assert by_name["quinoline"].atom_by_locant["1"].symbol == "N"
    assert by_name["isoquinoline"].atom_by_locant["2"].symbol == "N"


def test_opsin_resource_grammar_separates_tokens_from_production_gate():
    grammar = opsin_resource_grammar()
    gate = retained_fused_derivative_gate()
    tokens = grammar["retained_fused_tokens"]

    assert tokens["quinoline"]["parent_stems"] == ["quinolin"]
    assert "quinolino" in tokens["quinoline"]["fusion_stems"]
    assert "quinol" in tokens["quinoline"]["substituent_stems"]
    assert "quinoline" in gate.production_parent_names
    assert retained_fused_token_status("quinoline") == "production_safe"

    for parent in ("azulene", "1H-phenalene", "acenaphthylene", "fluoranthene", "pteridine"):
        assert parent in tokens
        assert parent in gate.audit_only_parent_names
        assert parent not in gate.production_parent_names
        assert retained_fused_token_status(parent) == "audit_only"


def test_opsin_resource_retained_fused_gate_is_derived_from_token_status():
    tokens = retained_fused_tokens()
    gate = retained_fused_derivative_gate()

    expected_production = {
        parent for parent, token in tokens.items()
        if token.derivative_status == "production_safe"
    }
    expected_audit_only = {
        parent for parent, token in tokens.items()
        if token.derivative_status == "audit_only"
    }

    assert gate.production_parent_names == expected_production
    assert gate.audit_only_parent_names == expected_audit_only


def test_opsin_resource_tokens_cover_retained_fused_template_stems():
    templates = retained_fused_graph_templates(include_disabled=True)

    for template in templates:
        token = retained_fused_token(template.name)
        if token is None:
            continue
        derivative_stems = {template.derivative_stem}
        if len(template.derivative_stem) > 3 and template.derivative_stem[1:3] == "H-":
            derivative_stems.add(template.derivative_stem[3:])
        assert derivative_stems & set(token.parent_stems)
        assert template.attached_prefix in token.fusion_stems
        if template.default_indicated_h:
            assert template.default_indicated_h == token.default_indicated_h
        if token.derivative_status == "production_safe":
            assert template.derivative_production_enabled
        else:
            assert not template.derivative_production_enabled


def test_opsin_resource_grammar_records_charge_and_heteroatom_tokens():
    grammar = opsin_resource_grammar()

    assert grammar["charge_suffixes"]["canonical"] == ["ium", "ide", "ylium", "uide"]
    assert grammar["charge_suffixes"]["accepted_spellings"]["uide"] == ["uide", "uid"]
    assert grammar["hetero_replacement_priority"]["prefix_order"][:9] == [
        "fluora",
        "chlora",
        "broma",
        "ioda",
        "oxa",
        "thia",
        "selena",
        "tellura",
        "aza",
    ]
    assert grammar["halogen_oxoacid_common_names"]["Cl"]["XO4"] == "perchloric acid"
    suffixes = oxoacid_ester_suffix_templates()
    assert suffixes["phosphate"]["neutral_monoester"] == "dihydrogen phosphate"
    assert suffixes["charge_normalized_halogen_peroxy_oxoester"]["Cl"] == "peroxychlorate"


def test_retained_fused_graph_templates_match_core_numbering_when_enabled_for_audit():
    cases = {
        "c1ccc2ccccc2c1": ("naphthalene", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "c1ccc2ncccc2c1": ("quinoline", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "c1ccc2cnccc2c1": ("isoquinoline", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "n1cccc2ncccc12": ("1,5-naphthyridine", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "n1cccc2cnccc12": ("1,6-naphthyridine", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "n1cccc2ccncc12": ("1,7-naphthyridine", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "n1cccc2cccnc12": ("1,8-naphthyridine", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "c1nccc2cnccc12": ("2,6-naphthyridine", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "c1nccc2ccncc12": ("2,7-naphthyridine", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "n1cncc2ccccc12": ("quinazoline", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "n1ccnc2ccccc12": ("quinoxaline", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "n1nccc2ccccc12": ("cinnoline", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "c1nncc2ccccc12": ("phthalazine", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "C1=CC=C2C=CC=CC=C12": ("azulene", {"1", "2", "3", "3a", "4", "5", "6", "7", "8", "8a"}),
        "C1C=CC2=CC=CC3=CC=CC1=C23": ("1H-phenalene", {"1", "2", "3", "3a", "4", "5", "6", "6a", "7", "8", "9", "9a", "9b"}),
        "C1=CC2=CC=CC3=CC=CC1=C23": ("acenaphthylene", {"1", "2", "2a", "3", "4", "5", "5a", "6", "7", "8", "8a", "8b"}),
        "C1=CC=C2C=CC=C3C4=CC=CC=C4C1=C23": ("fluoranthene", {"1", "2", "3", "3a", "4", "5", "6", "6a", "6b", "7", "8", "9", "10", "10a", "10b", "10c"}),
        "N1=CN=CC2=NC=CN=C12": ("pteridine", {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}),
        "N1C=NC2=CC=CC3=CC=CC1=C23": ("1H-perimidine", {"1", "2", "3", "3a", "4", "5", "6", "6a", "7", "8", "9", "9a", "9b"}),
        "C=1NC=C2C=CC=CC12": ("2H-isoindole", {"1", "2", "3", "3a", "4", "5", "6", "7", "7a"}),
        "C=1C=CN2C=CC=CC12": ("indolizine", {"1", "2", "3", "4", "5", "6", "7", "8", "8a"}),
    }

    for smiles, (expected_parent, expected_locants) in cases.items():
        mol = read_smiles(smiles)
        ring = find_ring_systems(mol)[0]
        matches = match_retained_fused_templates(mol, list(ring.atoms), include_disabled=True)
        matches = [match for match in matches if match.template.name == expected_parent]

        assert matches
        assert set(matches[0].atom_to_locant.values()) == expected_locants
        assert set(matches[0].atom_to_locant) == set(ring.atoms)


def test_retained_fused_template_matching_is_charge_exact():
    row = _naphthalene_graph_template_row()
    row["template"]["atoms"][0]["symbol"] = "N"
    row["template"]["atoms"][0]["charge"] = 1
    template = retained_fused_template_from_data(row)
    charged = template_molecule(template)
    neutral = template_molecule(template)
    neutral.atoms[0].charge = 0

    assert match_retained_fused_template(charged, set(charged.atoms), template) is not None
    assert match_retained_fused_template(neutral, set(neutral.atoms), template) is None


def test_requested_retained_fused_parents_are_pending_until_graph_verified():
    pending = set(pending_retained_fused_parent_names())

    assert {
        "pleiadene",
        "phenanthridine",
        "4H-quinolizine",
        "1H-pyrrolizine",
    } <= pending
    assert not (pending & {template.name for template in retained_fused_graph_templates(include_disabled=True)})


def test_fused_topology_route_keeps_disabled_retained_templates_out_of_production():
    mol = read_smiles("c1ccc2ccccc2c1")
    ring = find_ring_systems(mol)[0]

    live_route = classify_ring_topology_route(mol, set(ring.atoms))
    audit_route = classify_ring_topology_route(mol, set(ring.atoms), include_disabled_retained=True)

    assert live_route.kind == RingTopologyRouteKind.SYSTEMATIC_FUSED
    assert live_route.production_ready is False
    assert audit_route.kind == RingTopologyRouteKind.RETAINED_FUSED
    assert audit_route.production_ready is False
    assert audit_route.retained_matches[0].template.name == "naphthalene"


def test_general_retained_ring_recognizer_does_not_call_fused_template_matcher():
    assert not hasattr(retained_rules, "match_retained_fused_templates")


def test_fused_component_and_numbering_candidates_are_graph_bound():
    mol = read_smiles("c1ccc2ncccc2c1")
    ring = find_ring_systems(mol)[0]
    route = classify_ring_topology_route(mol, set(ring.atoms), include_disabled_retained=True)
    match = next(match for match in route.retained_matches if match.template.name == "quinoline")

    component = fused_component_from_retained_match(mol, match)
    numbering = fused_numbering_from_retained_match(match)

    assert component.name == "quinoline"
    assert component.source == "retained_fused_template"
    assert component.production_ready is False
    assert component.heteroatom_symbols == ("N",)
    assert numbering.audit_ok
    assert set(numbering.atom_to_locant) == set(ring.atoms)
    assert numbering.locant_to_atom["1"] in ring.atoms
    assert numbering.peripheral_locants == match.template.peripheral_atoms
    assert numbering.fusion_atom_locants == match.template.fusion_atoms
    assert numbering.heteroatom_locants == ("1",)
    assert numbering.orientation_source == "retained_template"


def test_stage6_fused_component_candidate_carries_descriptor_inputs():
    mol = read_smiles("c1ccc2ncccc2c1")
    ring = find_ring_systems(mol)[0]
    route = classify_ring_topology_route(mol, set(ring.atoms), include_disabled_retained=True)
    match = next(match for match in route.retained_matches if match.template.name == "quinoline")

    component = fused_component_from_retained_match(mol, match)

    assert component.component_locants == ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a")
    assert component.attached_prefix == "quinolino"
    assert component.derivative_stem == "quinolin"
    assert component.opsin_token_status == "production_safe"
    assert component.heteroatom_symbols == ("N",)
    assert component.heteroatom_count == 1
    assert component.heteroatom_variety == 1
    assert component.senior_heteroatom_count == 1
    assert component.fusion_sides[:3] == (
        ("a", ("1", "2")),
        ("b", ("2", "3")),
        ("c", ("3", "4")),
    )


def test_stage6_parent_side_letters_are_template_bond_backed():
    template = next(
        template
        for template in retained_fused_graph_templates(include_disabled=True)
        if template.name == "naphthalene"
    )

    sides = fused_parent_side_letters(template)

    assert sides[0] == ("a", ("1", "2"))
    assert ("e", ("4a", "5")) in sides
    assert all(tuple(sorted(locants)) in {tuple(sorted(bond.locants)) for bond in template.bonds} for _, locants in sides)


def test_stage6_fused_component_registry_uses_templates_and_opsin_tokens():
    registry = fused_component_registry(include_disabled=True)
    quinoline = registry.by_name["quinoline"]

    assert quinoline.component_id == "retained_fused:quinoline"
    assert quinoline.accepted_name == "quinoline"
    assert quinoline.fusion_prefix_name == "quinolino"
    assert quinoline.derivative_stem == "quinolin"
    assert quinoline.atom_locants == ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a")
    assert quinoline.fusion_side_letters[:3] == (
        ("a", ("1", "2")),
        ("b", ("2", "3")),
        ("c", ("3", "4")),
    )
    assert quinoline.ring_count == 2
    assert quinoline.ring_size_sequence == (6, 6)
    assert quinoline.heteroatom_symbols == ("N",)
    assert quinoline.retained_seniority_rank == 13
    assert quinoline.is_mancude
    assert quinoline.is_retained_parent_component
    assert quinoline.is_allowed_as_fusion_component
    assert quinoline.opsin_token_status == "production_safe"
    assert "quinoline" in quinoline.opsin_parseable_names

    indole = registry.by_name["1H-indole"]
    assert indole.opsin_token_status == "audit_only"
    assert not indole.is_allowed_as_fusion_component


def test_stage6_fused_component_registry_orders_allowed_parent_candidates():
    registry = fused_component_registry(include_disabled=True)
    candidates = registry.parent_component_candidates()

    assert candidates
    assert all(candidate.is_allowed_as_fusion_component for candidate in candidates)
    assert "1H-indole" not in {candidate.accepted_name for candidate in candidates}
    assert candidates == tuple(sorted(candidates, key=lambda candidate: candidate.parent_component_key))


def test_plain_fused_parent_does_not_create_bridged_fused_candidate():
    mol = read_smiles("c1ccc2ccccc2c1")
    ring = find_ring_systems(mol)[0]
    route = classify_ring_topology_route(mol, set(ring.atoms))

    assert route.kind == RingTopologyRouteKind.SYSTEMATIC_FUSED
    assert bridged_fused_candidates(mol, route) == ()


def test_spiro_component_reference_requires_a_complete_component_locant_map():
    incomplete = spiro_component_reference(
        "cyclopropane",
        {0: "1", 1: "2"},
        0,
        {0, 1, 2},
        source="unit-test",
    )
    complete = spiro_component_reference(
        "cyclopropane",
        {0: "1", 1: "2", 2: "3"},
        0,
        {0, 1, 2},
        source="unit-test",
    )

    assert not incomplete.audit_ok
    assert "spiro component locant map does not cover component atoms" in incomplete.audit_errors
    assert complete.audit_ok


def test_stage8_bridged_fused_candidate_binds_bridge_attachment_locants():
    template = next(
        template
        for template in retained_fused_graph_templates(include_disabled=True)
        if template.name == "naphthalene"
    )
    mol = template_molecule(template)
    mol.add_atom("C", 10)
    mol.add_bond(template.locants.index("2"), 10, order=1, idx=100)
    mol.add_bond(template.locants.index("7"), 10, order=1, idx=101)
    match = match_retained_fused_template(mol, set(range(10)), template)
    assert match is not None
    route = RingTopologyRoute(
        kind=RingTopologyRouteKind.BRIDGED_FUSED,
        atoms=frozenset(mol.atoms),
        topology=ring_system_topology(mol, set(mol.atoms)),
        retained_matches=(match,),
        reason="unit-test bridged fused route",
    )

    candidates = bridged_fused_candidates(mol, route)
    bridge_candidate = next(candidate for candidate in candidates if candidate.bridge_atoms == frozenset({10}))

    assert candidates
    assert bridge_candidate.bridge_attachment_locants == ("2", "7")
    assert bridge_candidate.bridge_length == 1


def test_stage9_spiro_component_reference_carries_primed_display_locants():
    reference = spiro_component_reference(
        "cyclopropane",
        {0: "1", 1: "2", 2: "3"},
        0,
        {0, 1, 2},
        source="unit-test",
        prime_count=1,
    )

    assert reference.audit_ok
    assert reference.atom_to_locant[0] == "1"
    assert reference.display_atom_to_locant[0] == "1'"


def test_charged_fused_template_gate_waits_for_neutral_parent_verification():
    mol = read_smiles("[nH+]1ccccc1")
    ring = find_ring_systems(mol)[0]

    gate = charged_fused_template_gate(
        "pyridin-1-ium",
        mol,
        set(ring.atoms),
        neutral_parent_verified=False,
    )

    assert gate.charged_atoms
    assert not gate.production_ready


def test_stage10_fused_ion_template_registry_is_data_backed_and_gated():
    registry = fused_ion_template_registry()
    quinoline_ium = registry.by_id["quinoline_ring_n_1_ium"]

    assert quinoline_ium.base_parent_name == "quinoline"
    assert quinoline_ium.allowed_charge_operation == "ring_n_ium"
    assert quinoline_ium.allowed_locants == ("1",)
    assert quinoline_ium.suffix_or_prefix_form == "ium"
    assert quinoline_ium.opsin_grammar_name == "quinolin-1-ium"
    assert quinoline_ium.opsin_compatible_spelling == "quinolin-1-ium"
    assert quinoline_ium.base_parent_token_status == "production_safe"
    assert quinoline_ium.production_ready
    assert {template.template_id for template in registry.production_templates()} == {
        "quinoline_ring_n_1_ium",
        "isoquinoline_ring_n_2_ium",
        "quinoline_ring_n_1_oxide",
        "isoquinoline_ring_n_2_oxide",
    }
    assert registry.for_parent("1H-indole")[0].suffix_or_prefix_form == "ide"
    assert registry.for_parent("indole")[0].suffix_or_prefix_form == "ide"
    assert registry.by_id["quinoline_ring_n_1_oxide"].opsin_grammar_name == "quinoline 1-oxide"


def test_stage10_retained_fused_ring_n_charge_uses_graph_operation_renderer():
    cases = {
        "[nH+]1cccc2ccccc12": ("quinolin-1-ium", "fused_ring_n_ium", "1"),
        "[nH+]1ccc2ccccc2c1": ("isoquinolin-2-ium", "fused_ring_n_ium", "2"),
    }

    for smiles, (expected, role, locant) in cases.items():
        analysis = analyze_smiles(smiles)
        assembly = [step for step in analysis.decisions if step.decision == "assembled component name"][-1]
        bindings = assembly.data["name_atom_bindings"]

        assert analysis.name == expected
        assert any(
            binding["role"] == role
            and binding["term"] == expected
            and binding["locants"] == [locant]
            for binding in bindings
        )
        assert not any(binding["role"] == "parent_charge" and binding["locants"] == [locant] for binding in bindings)


def test_stage10_retained_fused_ion_derivatives_keep_substituents_on_candidate_parent():
    cases = {
        "C[n+]1cccc2ccccc12": ("1-methylquinolin-1-ium", "fused_ring_n_ium", "quinolin-1-ium"),
        "C[n+]1ccc2ccccc2c1": ("2-methylisoquinolin-2-ium", "fused_ring_n_ium", "isoquinolin-2-ium"),
        "Cc1cc[n+]([O-])c2ccccc12": ("4-methylquinoline 1-oxide", "fused_n_oxide", "quinoline 1-oxide"),
        "Cc1c[n+]([O-])cc2ccccc12": ("4-methylisoquinoline 2-oxide", "fused_n_oxide", "isoquinoline 2-oxide"),
    }

    for smiles, (expected, role, fused_term) in cases.items():
        analysis = analyze_smiles(smiles)
        assembly = [step for step in analysis.decisions if step.decision == "assembled component name"][-1]
        bindings = assembly.data["name_atom_bindings"]

        assert analysis.name == expected
        assert any(binding["role"] == "substituent" and binding["term"] == "methyl" for binding in bindings)
        assert any(binding["role"] == role and binding["term"] == fused_term for binding in bindings)
        assert not any(binding["role"] == "parent_charge" for binding in bindings)
        if role == "fused_n_oxide":
            assert not any(binding["role"] == "substituent" and binding["term"] == "oxido" for binding in bindings)


def test_stage10_fused_ion_candidates_are_graph_operations_before_rendering():
    parts = AssemblyParts(
        parent_length=10,
        is_ring=True,
        retained_name="quinoline",
        substituents=[SubstituentItem(name="oxido", locants=["1"], atom_ids={0, 1}, bond_ids={99})],
        parent_charges=[ParentChargeItem(locant="1", symbol="N", charge=1, atom_id=1)],
        parent_atom_symbols_by_locant={"1": "N"},
        parent_atom_charges_by_locant={"1": 1},
    )

    candidates = fused_ion_operation_candidates(parts)

    assert len(candidates) == 2
    candidate = candidates[0]
    assert candidate.operation == "ring_n_oxide"
    assert candidate.locants == ("1",)
    assert candidate.represented_atom_ids == frozenset({0, 1})
    assert candidate.represented_bond_ids == frozenset({99})
    assert candidate.production_ready
    assert candidate.rendered_name == "quinoline 1-oxide"


def test_stage10_retained_fused_n_oxide_uses_generic_opsin_grammar_renderer():
    cases = {
        "[O-][n+]1cccc2ccccc12": "quinoline 1-oxide",
        "[O-][n+]1ccc2ccccc2c1": "isoquinoline 2-oxide",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_stage10_retained_fused_n_oxide_binding_consumes_charge_pair():
    analysis = analyze_smiles("[O-][n+]1cccc2ccccc12")
    assembly = [step for step in analysis.decisions if step.decision == "assembled component name"][-1]
    bindings = assembly.data["name_atom_bindings"]

    assert analysis.name == "quinoline 1-oxide"
    assert any(
        binding["role"] == "fused_n_oxide"
        and binding["term"] == "quinoline 1-oxide"
        and binding["locants"] == ["1"]
        and binding["atoms"] == [0, 1]
        for binding in bindings
    )
    assert not any(binding["role"] == "substituent" and binding["term"] == "oxido" for binding in bindings)
    assert not any(binding["role"] == "parent_charge" and binding["locants"] == ["1"] for binding in bindings)


def test_stage6_to_10_emission_examples_are_generic_opsin_grammar_targets():
    examples = fused_emission_examples()

    assert examples.name_policy == "generic_opsin_grammar"
    assert "furo[3,2-b]thieno[2,3-e]pyridine" in examples.stages["stage_6_systematic_fused_descriptor"]
    assert "1H-cyclopenta[l]phenanthrene" in examples.stages["stage_7_fused_layout_numbering"]
    assert "10,5-[2,3]furanobenzo[g]quinoline" in examples.stages["stage_8_bridged_fused"]
    assert "1'H-spiro[imidazolidine-4,2'-quinoxaline]" in examples.stages["stage_9_spiro_wrapper"]
    assert set(examples.stages["stage_10_charged_fused_heteroaromatics"]) == {
        "quinolin-1-ium",
        "isoquinolin-2-ium",
        "quinoline 1-oxide",
        "indol-1-ide",
        "carbazol-9-ide",
    }


def test_retained_fused_derivative_gate_uses_template_locants_for_opsin_safe_classes():
    cases = {
        "Cc1ccc2ncccc2c1": "6-methylquinoline",
        "COc1ccc2ncccc2c1": "6-methoxyquinoline",
        "Clc1ccc2ncccc2c1": "6-chloroquinoline",
        "Oc1ccc2ncccc2c1": "quinolin-6-ol",
        "Nc1ccc2ncccc2c1": "quinolin-6-amine",
        "N#Cc1ccc2ncccc2c1": "quinoline-6-carbonitrile",
        "O=Cc1ccc2ncccc2c1": "quinoline-6-carbaldehyde",
        "O=C(N)c1ccc2ncccc2c1": "quinoline-6-carboxamide",
        "O=C(O)c1ccc2ncccc2c1": "quinoline-6-carboxylic acid",
        "Cc1ccc2cnccc2c1": "6-methylisoquinoline",
        "n1cccc2ncccc12": "1,5-naphthyridine",
        "n1cccc2cnccc12": "1,6-naphthyridine",
        "n1cccc2ccncc12": "1,7-naphthyridine",
        "n1cccc2cccnc12": "1,8-naphthyridine",
        "c1nccc2cnccc12": "2,6-naphthyridine",
        "c1nccc2ccncc12": "2,7-naphthyridine",
        "Cc1nccc2cnccc12": "1-methyl-2,6-naphthyridine",
        "n1cncc2ccccc12": "quinazoline",
        "n1ccnc2ccccc12": "quinoxaline",
        "n1nccc2ccccc12": "cinnoline",
        "c1nncc2ccccc12": "phthalazine",
        "Cc1ncnc2ccccc12": "4-methylquinazoline",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_retained_fused_graph_template_rejects_incomplete_locant_maps():
    row = _naphthalene_graph_template_row()
    row["template"]["atoms"] = row["template"]["atoms"][:-1]

    with pytest.raises(ValueError, match="atom locants do not match"):
        retained_fused_template_from_data(row)


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

    no_group = ParentCandidate.build(
        [0, 1],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )
    one_group = ParentCandidate.build(
        [0, 1],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=1,
        mol=mol,
    )
    two_groups = ParentCandidate.build(
        [0, 1, 2],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=2,
        mol=mol,
    )
    n_parent = ParentCandidate.build(
        [3],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )
    si_parent = ParentCandidate.build(
        [4],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )
    carbon_ring = ParentCandidate.build(
        [10, 11, 12, 13],
        is_ring=True,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
        ring_count=1,
    )
    carbon_chain = ParentCandidate.build(
        [10, 11, 12, 13],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )
    o_ring = ParentCandidate.build(
        [5, 10, 11],
        is_ring=True,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
        ring_count=1,
    )
    p_ring = ParentCandidate.build(
        [6, 10, 11],
        is_ring=True,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
        ring_count=1,
    )
    bicycle = ParentCandidate.build(
        [10, 11, 12, 13],
        is_ring=True,
        is_bicycle=True,
        is_spiro=False,
        is_polycycle=False,
        xyz=(1, 1, 0),
        principal_groups_count=0,
        mol=mol,
        ring_count=2,
    )
    monocycle = ParentCandidate.build(
        [10, 11, 12, 13],
        is_ring=True,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
        ring_count=1,
    )
    piperazine_like = ParentCandidate.build(
        [7, 8, 10, 11, 12, 13],
        is_ring=True,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
        ring_count=1,
    )
    oxazinane_like = ParentCandidate.build(
        [7, 9, 10, 11, 12, 13],
        is_ring=True,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
        ring_count=1,
    )
    shorter_chain = ParentCandidate.build(
        [10, 11, 12],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )
    longer_chain = ParentCandidate.build(
        [10, 11, 12, 13],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )
    saturated_chain = ParentCandidate.build(
        [10, 11, 12],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )
    unsaturated_chain = ParentCandidate.build(
        [0, 1, 2],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        principal_groups_count=0,
        mol=mol,
    )

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


@pytest.mark.xfail(
    reason=(
        "Output depends on RDKit's aromaticity perception. On rdkit 2026.x "
        "(used in CI) the namer emits the expected "
        "nona-1(6),2,4-trien-8-one; on older rdkit 2025.x it emits "
        "nona-1,3,5-trien-8-one. strict=False so the test passes either "
        "way until the underlying rdkit-version sensitivity is addressed."
    ),
    strict=False,
)
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
        "C[C@H]1C[NH+](CCN1c2[nH]c3ccccc3n2)C": "2-((2S)-2,4-dimethylpiperazin-4-ium-1-yl)-1H-benzimidazole",
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
        build_ring_numbering(
            "bicyclo", proof.descriptor_numbers, path, topology.edges, mol, substituent_attachment_atoms={1}
        )
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
        apply_retained_parent_ide("5-(ammoniomethyl)-1,2,3-triazole-4-carbonitrile", "1,2,3-triazole", {"1": {"C"}})
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


def test_ring_parent_nitrogen_zwitterion_stack_handles_cumulative_charge_sites():
    mol = Molecule()
    for idx, charge in enumerate((-1, 1, 0, -1, 1, 0)):
        mol.add_atom("N" if idx in {0, 1, 3, 4} else "C", idx=idx, charge=charge)
    for idx, (first, second, order) in enumerate(
        [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1)],
        start=1,
    ):
        mol.add_bond(first, second, order=order, idx=idx)
    numbered_path = [0, 1, 2, 3, 4, 5]
    locants = {atom_idx: str(idx + 1) for idx, atom_idx in enumerate(numbered_path)}

    updated = apply_ring_parent_nitrogen_zwitterion_stack(
        "1,2,4,5-tetrazacyclohexa-1,3,5-trien-2,5-ium",
        mol,
        numbered_path,
        lambda atom_idx: locants[atom_idx],
    )

    assert updated == "1,2,4,5-tetrazacyclohexa-1,3,5-trien-1,4-ide-2,5-ium"


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


def test_von_baeyer_candidate_search_builds_graph_ranked_descriptor():
    mol = Molecule()
    for idx in range(1, 8):
        mol.add_atom("C", idx=idx)
    edges = {
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 1),
        (1, 7),
        (7, 4),
        (2, 6),
    }
    for idx, (first, second) in enumerate(sorted(edges), start=1):
        mol.add_bond(first, second, order=1, idx=idx)

    candidates = find_von_baeyer_candidates(mol, set(range(1, 8)), edges)

    assert candidates
    assert candidates[0].descriptor == "tricyclo[2.2.1.0^{2,6}]"
    assert candidates[0].numbering.audit_ok
    assert candidates[0].numbering.atom_to_locant[2] == 2
    assert candidates[0].numbering.atom_to_locant[6] == 6


def test_von_baeyer_audit_accepts_high_cycle_count_prefixes():
    descriptor = "heptacyclo[2.2.1.0^{2,6}.0^{2,7}.0^{3,5}.0^{3,7}.0^{5,7}]"
    path = tuple(range(1, 8))
    base_edges = {
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (1, 6),
        (1, 7),
        (4, 7),
    }
    extra_edges = {(2, 6), (2, 7), (3, 5), (3, 7), (5, 7)}

    audit = audit_von_baeyer_descriptor(descriptor, path, base_edges | extra_edges)

    assert audit.audit_ok


def test_von_baeyer_candidate_search_can_render_heptacyclo_candidate():
    mol = Molecule()
    for idx in range(1, 8):
        mol.add_atom("C", idx=idx)
    edges = {
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (1, 6),
        (1, 7),
        (4, 7),
        (2, 6),
        (2, 7),
        (3, 5),
        (3, 7),
        (5, 7),
    }
    for idx, (first, second) in enumerate(sorted(edges), start=1):
        mol.add_bond(first, second, order=1, idx=idx)

    candidates = find_von_baeyer_candidates(mol, set(range(1, 8)), edges)

    assert candidates
    assert candidates[0].descriptor.startswith("heptacyclo[")
    assert candidates[0].numbering.audit_ok


def test_von_baeyer_secondary_bridge_classifier_keeps_dependent_bridges_after_independent_bridges():
    atom_set = frozenset(range(1, 10))
    edge_set = frozenset(
        {
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 1),
            (1, 7),
            (7, 4),
            (2, 8),
            (8, 9),
            (6, 9),
            (4, 8),
        }
    )
    primary_atoms = {1, 2, 3, 4, 5, 6, 7}
    primary_edges = {
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 1),
        (1, 7),
        (7, 4),
    }

    bridges = _classify_secondary_bridges(
        atom_set=atom_set,
        edge_set=edge_set,
        primary_atoms=primary_atoms,
        remaining_edges=frozenset(edge_set - primary_edges),
    )

    assert bridges is not None
    assert [(bridge.length, bridge.dependent) for bridge in bridges] == [(2, False), (0, True)]


def test_polycycle_without_descriptor_fails_closed():
    parts = AssemblyParts(parent_length=6, is_polycycle=True)

    with pytest.raises(ValueError, match="polycyclic parent has no audited descriptor"):
        assemble_name(parts)


def test_von_baeyer_descriptor_audit_reconstructs_source_edges():
    mol = read_smiles("C1C2C1C13COC21CO3")
    ring = next(system for system in find_ring_systems(mol, set()) if system.polycycle_descriptor)
    audit = audit_von_baeyer_descriptor(
        ring.polycycle_descriptor,
        ring.paths[0],
        {(a, b) for a in ring.atoms for b in mol.get_neighbors(a) if b in ring.atoms and a < b},
    )

    assert audit.audit_ok
    assert ring.ring_parent is not None
    assert ring.ring_parent.numbering_candidates
    assert ring.ring_parent.audit_ok


def test_von_baeyer_descriptor_audit_rejects_wrong_locants():
    mol = read_smiles("C1C2C1C13COC21CO3")
    ring = next(system for system in find_ring_systems(mol, set()) if system.polycycle_descriptor)
    bad_descriptor = ring.polycycle_descriptor.replace("0^{1,5}", "0^{1,4}")
    audit = audit_von_baeyer_descriptor(
        bad_descriptor,
        ring.paths[0],
        {(a, b) for a in ring.atoms for b in mol.get_neighbors(a) if b in ring.atoms and a < b},
    )

    assert not audit.audit_ok


def test_von_baeyer_ring_parent_requires_audited_numbering_candidates():
    with pytest.raises(ValueError, match="requires audited numbering"):
        RingParent.from_paths(
            kind="polycycle",
            atoms={0, 1, 2},
            descriptor="tricyclo[1.1.0.0^{1,3}]butane",
            paths=[[0, 1, 2]],
        )


def test_parent_pipeline_uses_audited_von_baeyer_locant_maps():
    mol = read_smiles("C1C2C1C13COC21CO3")
    ring = next(system for system in find_ring_systems(mol, set()) if system.polycycle_descriptor)
    selection = ParentSelection(
        paths=ring.paths,
        is_ring=True,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=True,
        xyz=(0, 0, 0),
        polycycle_descriptor=ring.polycycle_descriptor,
        ring_parent=ring.ring_parent,
    )

    plan = build_parent_assembly_plan(mol, selection, NamingIntent.component([]), {}, None, None)

    assert ring.ring_parent is not None
    assert ring.ring_parent.selected_numbering is not None
    assert plan.locant_map in [numbering.locant_map for numbering in ring.ring_parent.numbering_candidates]
    assert plan.parts.parent_atom_ids_by_locant["1"] == next(
        atom for atom, locant in plan.locant_map.items() if locant == "1"
    )


def test_high_risk_polycycle_audit_fails_closed_without_proof_candidate():
    result = NamingEngine().run(NamingRequest(smiles="C1Oc2c3c(c(c4c2C4)O1)C3"))

    assert result.name == ""
    assert result.error is not None

    mol = read_smiles("C1Oc2c3c(c(c4c2C4)O1)C3")
    rings = [system for system in find_ring_systems(mol, set()) if system.is_polycycle]
    assert rings
    assert not any(ring.polycycle_descriptor for ring in rings)


def test_charge_separated_sulfonium_ylide_requires_single_bond():
    assert name_smiles("C[S+](C)[CH-]C") == "1-(dimethylsulfaniumyl)ethan-1-ide"
    assert name_smiles("C[SH+](C)=[C-]c1ccccc1") == "((dimethyl-lambda^4-sulfanylidene)methyl)benzene"


def test_sulfur_ylide_resonance_compare_accepts_lambda_fallback_graph():
    assert equivalent_smiles("C[SH+](C)=[C-]c1ccccc1", "CS(C)=CC1=CC=CC=C1")
    assert equivalent_smiles("C[S+](C)[CH-]C", "CS(C)=CC")
    assert not equivalent_smiles("C[S+](C)[CH-]C", "CCSC")


def test_charge_separated_diazo_hypervalent_ring_templates_do_not_use_imino_aminium():
    assert name_smiles("[N-]=[N+]=P1=CC=CC=C1") == "1-diazo-1lambda^5-phosphacyclohexa-1,3,5-triene"
    assert name_smiles("[N-]=[N+]=[Se]1C=CN=C1") == "1-diazo-3-aza-1lambda^4-selenacyclopenta-2,4-diene"
    assert name_smiles("[N-]=[N+]=S1C=CC1") == "1-diazo-1lambda^4-thiacyclobut-2-ene"


def test_terminal_chalcogen_imides_render_as_imino_prefixes():
    assert name_smiles("N=S1C=CC1") == "1-imino-1lambda^4-thiacyclobut-2-ene"
    assert name_smiles("N=[Se]1C=CC=C1") == "1-imino-1lambda^4-selenacyclopenta-2,4-diene"


def test_sulfur_imide_substituents_preserve_double_bonded_nitrogen():
    assert name_smiles("N=S(Cl)CF") == "((chloro)(imino)sulfanyl)fluoromethane"
    assert (
        name_smiles("CSC(=O)S(C)=N")
        == "1-((imino)(methyl)sulfanyl)-1-(methylsulfanyl)methanone"
    )


def test_sulfonimidoyl_substituents_keep_imino_n_ligand():
    assert (
        name_smiles("CC(C)N=S(C)(=O)c1ccc(N)cc1")
        == "4-(N-propan-2-yl-S-methylsulfonimidoyl)benzen-1-amine"
    )
    assert (
        name_smiles("CN=S(=O)(CC(C)N)NOC")
        == "1-(N-methyl-S-methoxyaminosulfonimidoyl)propan-2-amine"
    )


def test_cyclic_sulfur_imide_substituent_keeps_ring_ligand_together():
    assert (
        name_smiles("CC(C)(C)OC(=O)N=S1(=O)CCC1")
        == "tert-butyl ((1-oxo-1lambda^6-thietan-1-ylidene)amino)formate"
    )


def test_terminal_s_minus_uses_thiolate_role():
    assert name_smiles("[S-][n+]1ccc(-c2ccncc2)cc1") == "4-(pyridin-4-yl)pyridin-1-ium-1-thiolate"


def test_terminal_selenium_anion_substituent_preserves_charge():
    assert name_smiles("[Se-]C1=CC2(C=CN1)CC[NH2+]CC2") == "2-selenido-3,9-diazaspiro[5.5]undeca-1,4-dien-9-ium"


def test_charge_separated_terminal_n3_renders_as_azido_role():
    cases = {
        "C=CC(=O)NN=[N+]=[N-]": "N-azidoprop-2-enamide",
        "[N-]=[N+]=Nn1cncn1": "1-azido-1H-1,2,4-triazole",
        "CCP(CC)(CC)(CC)N=[N+]=[N-]": "1-((azido)triethylphosphanyl)ethane",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_azine_retained_and_simple_ring_sides_are_graph_bound():
    assert (
        name_smiles("C1=CC(C=NN=C2SCCS2)C=C1")
        == "cyclopenta-2,4-dien-1-carbaldehyde 1,3-dithiacyclopentan-2-ylidenehydrazone"
    )


def test_charged_sulfur_substituent_preserves_sulfanium_center():
    assert name_smiles("C[SH+]C(C)C(=O)[O-]") == "2-(methylsulfaniumyl)propanoate"


def test_positive_oxygen_parent_charge_uses_ium_suffix():
    cases = {
        "[O-]c1c[o+]co1": "1,3-dioxacyclopenta-1,4-dien-1-ium-4-olate",
        "[O-][o+]1ccc(=S)c2ccccc21": "2-oxido-5-thioxo-2-oxabicyclo[4.4.0]deca-1(10),3,6,8-tetraen-2-ium",
    }
    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_n_hydroxyurea_roles_avoid_carbamic_acid_fallback():
    cases = {
        "O=C(NO)NC1CCCCCCC1": "N-cyclooctyl-N'-hydroxyurea",
        "CCN(O)C(=O)NC1CCCCC1": "N-cyclohexyl-N'-ethyl-N'-hydroxyurea",
        "CC(C)(C)CCNC(=O)NO": "N-(3,3-dimethylbutyl)-N'-hydroxyurea",
        "CC(C)(C)CNC(=O)NO": "N-(2,2-dimethylpropyl)-N'-hydroxyurea",
    }
    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_simple_rooted_carbanion_substituent_preserves_charge():
    assert name_smiles("CC[CH-][n+]1cccc(C)c1") == "1-ethylmethanidyl-3-methylpyridin-1-ium"


def test_cyclic_peroxy_esters_render_as_oxo_dioxacycles():
    cases = {
        "O=C1OOCCCCC1C1CCCCCCC1": "4-cyclooctyl-1,2-dioxacyclooctan-3-one",
        "O=C1CCCCCCC(=O)OOCC1": "1,2-dioxacyclododecane-3,10-dione",
        "Cc1ccc2c(c1)C=CC(=O)OO2": "9-methyl-2,3-dioxabicyclo[5.4.0]undeca-1(7),5,8,10-tetraen-4-one",
        "CC(C)=CCC/C(C)=C1\\OOC1=O": "(4Z)-4-(6-methylhept-5-en-2-ylidene)-1,2-dioxacyclobutan-3-one",
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
