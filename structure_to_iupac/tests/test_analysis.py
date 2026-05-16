from structure_to_iupac import OperationClass, TracePhase, analyze_smiles, name_smiles
from structure_to_iupac.assembler import AssemblyParts, ParentChargeItem, SubstituentItem, UnsaturationItem, assemble_name
from structure_to_iupac.chains import find_all_carbon_paths, find_ring_systems
from structure_to_iupac.functional_groups import PERCEPTION_DETECTORS, register_group_detector
from structure_to_iupac.ionic_naming import apply_parent_charge_names, parent_charge_sites
from structure_to_iupac.locants import as_display_locant, coerce_display_numbering, locant_text, parse_locant
from structure_to_iupac.name_postprocessing import apply_connection_boundary_postprocessing
from structure_to_iupac.namer import read_smiles
from structure_to_iupac.parent_selection import ParentCandidate, ParentSelection, select_principal_parent
from structure_to_iupac.perception import PerceivedGroup, perceive_groups
from structure_to_iupac.ring_systems import ring_system_fragment
from structure_to_iupac.heteroatom_substituent_specs import ligand_prefix, unsubstituted_prefix


def test_name_smiles_stays_plain_fast_api():
    assert name_smiles("CCO") == "ethanol"


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
    assert selection.score_tuple


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
        "OCC1CC11C2CC1C2": "1-(spiro[cyclopropane-2,2'-(bicyclo[1.1.1]pentane)]-1-yl)methanol",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


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
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


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
