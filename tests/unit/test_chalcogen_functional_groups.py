"""Regression and feature tests for role-aware chalcogen functional groups."""

from itertools import product

import pytest
from rdkit import Chem

import openclatura as oc
from openclatura.chalcogen_roles import (
    Chalcogen,
    ChalcogenLigandRole,
    DerivativeKind,
    FunctionalFamily,
    UnsupportedChalcogenNomenclatureError,
    classify_chalcogen_ligand,
)
from openclatura.charge_pair_roles import NitrogenChalcogenideKind, charge_pair_roles
from openclatura.graph_io import read_smiles
from openclatura.perception import perceive_groups

VALIDATED_CHALCOGENS = (
    Chalcogen.OXYGEN,
    Chalcogen.SULFUR,
    Chalcogen.SELENIUM,
    Chalcogen.TELLURIUM,
)
SMILES_ATOM = {
    Chalcogen.OXYGEN: "O",
    Chalcogen.SULFUR: "S",
    Chalcogen.SELENIUM: "[Se]",
    Chalcogen.TELLURIUM: "[Te]",
}
SMILES_HYDROGENATED_ATOM = {
    Chalcogen.OXYGEN: "O",
    Chalcogen.SULFUR: "[SH]",
    Chalcogen.SELENIUM: "[SeH]",
    Chalcogen.TELLURIUM: "[TeH]",
}


def _descriptor(smiles: str, family: FunctionalFamily, derivative: DerivativeKind):
    matches = [
        group.descriptor
        for group in perceive_groups(read_smiles(smiles))
        if group.descriptor is not None
        and group.descriptor.family is family
        and group.descriptor.derivative is derivative
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("smiles", ["CC(=O)OC(C)=O", "O=C(C)OC(C)=O"])
def test_anhydride_perception_is_invariant_to_atom_zero(smiles: str):
    groups = perceive_groups(read_smiles(smiles))

    assert [group.key for group in groups].count("anhydride") == 1


@pytest.mark.parametrize(
    ("smiles", "expected_role"),
    [
        ("C=O", ChalcogenLigandRole.DOUBLE_BONDED),
        ("CO", ChalcogenLigandRole.HYDROGEN_BEARING),
        ("C[O-]", ChalcogenLigandRole.ANIONIC),
        ("COC", ChalcogenLigandRole.CARBON_LINK),
        ("CON", ChalcogenLigandRole.HETEROATOM_LINK),
        ("COO", ChalcogenLigandRole.CHALCOGEN_LINK),
        ("C[SeH]", ChalcogenLigandRole.HYDROGEN_BEARING),
    ],
)
def test_shared_chalcogen_ligand_classifier(smiles: str, expected_role: ChalcogenLigandRole):
    mol = read_smiles(smiles)
    ligand = classify_chalcogen_ligand(mol, 0, 1)

    assert ligand is not None
    assert ligand.role is expected_role


def test_neutral_chalcogen_radical_is_not_hydrogen_bearing():
    mol = read_smiles("C[Se]")

    assert mol.atoms[1].radical_electrons == 1
    assert classify_chalcogen_ligand(mol, 0, 1) is None


@pytest.mark.parametrize(
    ("smiles", "expected_key", "expected_name"),
    [
        ("C[SeH]", "selenol", "methaneselenol"),
        ("C[TeH]", "tellurol", "methanetellurol"),
        ("C[Se-]", "selenolate", "methaneselenolate"),
        ("C[Te-]", "tellurolate", "methanetellurolate"),
        ("CC(=S)C", "thioketone", "propane-2-thione"),
        ("CC(=[Se])C", "selenoketone", "propane-2-selone"),
        ("CC(=[Te])C", "telluroketone", "propane-2-tellone"),
        ("CC(=[Se])", "selenoaldehyde", "ethaneselenal"),
        ("CC(=[Te])", "telluroaldehyde", "ethanetellanal"),
    ],
)
def test_simple_chalcogen_analogues_are_named_from_graph_roles(smiles: str, expected_key: str, expected_name: str):
    import openclatura as oc

    mol = read_smiles(smiles)
    groups = perceive_groups(mol)
    matching = [group for group in groups if group.key == expected_key]

    assert len(matching) == 1
    assert matching[0].descriptor is not None
    assert oc.name(smiles).name == expected_name


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("COC", "methoxymethane"),
        ("CSC", "dimethylsulfane"),
        ("C[Se]C", "dimethylselane"),
        ("C[Te]C", "dimethyltellane"),
    ],
)
def test_ether_analogues_reuse_generic_heteroatom_subgraph_naming(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=S)O", "ethanethioic O-acid"),
        ("CC(=O)[SH]", "ethanethioic S-acid"),
        ("CC(=S)[SH]", "ethanedithioic acid"),
        ("CC(=[Se])O", "ethaneselenoic O-acid"),
        ("CC(=O)[SeH]", "ethaneselenoic Se-acid"),
        ("CC(=[Se])[SeH]", "ethanediselenoic acid"),
        ("CC(=[Te])O", "ethanetelluroic O-acid"),
        ("CC(=O)[TeH]", "ethanetelluroic Te-acid"),
        ("CC(=[Te])[TeH]", "ethaneditelluroic acid"),
        ("CC(=S)[SeH]", "ethaneselenothioic Se-acid"),
        ("CC(=[Se])[SH]", "ethaneselenothioic S-acid"),
        ("CC(=[Te])[SeH]", "ethaneselenotelluroic Se-acid"),
    ],
)
def test_mixed_chalcogen_acids_preserve_both_sites(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=O)SC", "S-methyl ethanethioate"),
        ("CC(=S)OC", "O-methyl ethanethioate"),
        ("CC(=S)SC", "S-methyl ethanedithioate"),
        ("CC(=[Se])OC", "O-methyl ethaneselenoate"),
        ("CC(=O)[Se]C", "Se-methyl ethaneselenoate"),
        ("CC(=[Te])N", "ethanetelluroamide"),
        ("CC(=[Se])NN", "ethaneselenohydrazide"),
        ("NC(=[Se])N", "selenourea"),
        ("NC(=[Te])N", "tellurourea"),
        ("CC(=[Se])Cl", "ethaneselenoyl chloride"),
    ],
)
def test_acyl_derivatives_reuse_the_role_aware_center(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("COO", "methaneperoxol"),
        ("CSO", "methane-SO-thioperoxol"),
        ("C[Se]O", "methane-SeO-selenoperoxol"),
        ("C[Te]O", "methane-TeO-telluroperoxol"),
        ("CO[SH]", "methane-OS-thioperoxol"),
        ("CO[SeH]", "methane-OSe-selenoperoxol"),
        ("CO[TeH]", "methane-OTe-telluroperoxol"),
        ("CS[SH]", "methanedithioperoxol"),
        ("C[Se][SH]", "methane-SeS-selenothioperoxol"),
        ("C[Te][SH]", "methane-TeS-tellurothioperoxol"),
        ("CS[SeH]", "methane-SSe-selenothioperoxol"),
        ("CS[TeH]", "methane-STe-tellurothioperoxol"),
        ("C[Se][SeH]", "methanediselenoperoxol"),
        ("C[Te][SeH]", "methane-TeSe-selenotelluroperoxol"),
        ("C[Se][TeH]", "methane-SeTe-selenotelluroperoxol"),
        ("C[Te][TeH]", "methaneditelluroperoxol"),
    ],
)
def test_peroxol_table_is_order_sensitive(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=O)OO", "ethaneperoxoic acid"),
        ("CC(=S)OO", "ethaneperoxothioic acid"),
        ("CC(=O)SO", "ethane(thioperoxoic) SO-acid"),
        ("CC(=O)O[SH]", "ethane(thioperoxoic) OS-acid"),
        ("CC(=[Se])S[TeH]", "ethaneseleno(tellurothioperoxoic) STe-acid"),
        ("CC(=O)OOC", "methyl ethaneperoxoate"),
        ("CC(=S)OOC", "OO-methyl ethaneperoxothioate"),
        ("CC(=O)SOC", "SO-methyl ethane(thioperoxoate)"),
        ("CC(=O)OSC", "OS-methyl ethane(thioperoxoate)"),
    ],
)
def test_peroxy_acyl_paths_preserve_orientation(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=S)OC(C)=O", "acetic ethanethioic anhydride"),
        ("CC(=S)OC(C)=S", "ethanethioic anhydride"),
        ("CC(=O)SC(C)=O", "acetic thioanhydride"),
        ("CC(=O)[Se]C(C)=O", "acetic selenoanhydride"),
        ("CC(=O)OOC(C)=O", "acetic peroxyanhydride"),
        ("CC(=O)SOC(C)=O", "acetic thioperoxyanhydride"),
        ("CC(=O)OSC(C)=O", "acetic thioperoxyanhydride"),
        ("CC(=O)SSC(C)=O", "acetic dithioperoxyanhydride"),
        ("CCC(=O)SOC(C)=O", "O-acetic S-propanoic thioperoxyanhydride"),
    ],
)
def test_anhydride_bridges_and_acyl_sites_are_independent(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CS(=O)(=O)O", "methanesulfonic acid"),
        ("CS(=O)O", "methanesulfinic acid"),
        ("C[Se](=O)(=O)O", "methaneselenonic acid"),
        ("C[Se](=O)O", "methaneseleninic acid"),
        ("C[Te](=O)(=O)O", "methanetelluronic acid"),
        ("C[Te](=O)O", "methanetellurinic acid"),
        ("CS(=S)(=O)O", "methanesulfonothioic O-acid"),
        ("CS(=O)(=O)S", "methanesulfonothioic S-acid"),
        ("C[Se](=S)(=O)O", "methaneselenonothioic O-acid"),
        ("CS(=O)(=O)OC", "methyl methanesulfonate"),
        ("CS(=O)(=O)N", "methanesulfonamide"),
        ("CS(=O)(=O)NN", "methanesulfonohydrazide"),
        ("CS(=O)(=O)N(N)C", "N-methylmethanesulfonohydrazide"),
        ("CS(=S)(=O)NN", "methanesulfonothiohydrazide"),
        ("C[Se](=O)(=O)NN", "methaneselenonohydrazide"),
        ("C[Te](=O)NN", "methanetellurinohydrazide"),
        ("CS(=O)(=O)Cl", "methanesulfonyl chloride"),
    ],
)
def test_central_chalcogen_acids_and_derivatives_share_one_topology(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


def test_central_hydrazide_locants_follow_the_functional_center_after_atom_renumbering():
    smiles = "CS(=O)(=O)N(C)NCC"
    expected = "N'-ethyl-N-methylmethanesulfonohydrazide"
    mol = Chem.MolFromSmiles(smiles)
    orders = (
        list(range(mol.GetNumAtoms())),
        list(reversed(range(mol.GetNumAtoms()))),
        list(range(1, mol.GetNumAtoms())) + [0],
    )

    assert {oc.name_mol(Chem.RenumberAtoms(mol, order)).name for order in orders} == {expected}
    if oc.opsin_available():
        check = oc.verify_with_opsin(expected, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=O)Sc1ccc(Cl)c(Cl)c1", "S-(3,4-dichlorophenyl) ethanethioate"),
        ("CC(=O)[Se]c1ccc(Cl)c(Cl)c1", "Se-(3,4-dichlorophenyl) ethaneselenoate"),
        ("CC(=O)[Te]c1ccc(Cl)c(Cl)c1", "Te-(3,4-dichlorophenyl) ethanetelluroate"),
    ],
)
def test_complex_chalcogen_front_modifiers_are_structurally_enclosed(smiles: str, expected: str):
    assert oc.name(smiles).name == expected
    if oc.opsin_available():
        check = oc.verify_with_opsin(expected, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CS(=O)(=O)OO", "methanesulfonoperoxoic acid"),
        ("CS(=O)(=O)SO", "methanesulfono(thioperoxoic) SO-acid"),
        ("CS(=O)(=O)O[SH]", "methanesulfono(thioperoxoic) OS-acid"),
        ("CS(=O)(=O)OOC", "methyl methanesulfonoperoxoate"),
        ("CS(=O)(=O)SOC", "SO-methyl methanesulfono(thioperoxoate)"),
    ],
)
def test_central_chalcogen_peroxy_paths_preserve_orientation(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "kind", "expected"),
    [
        ("C[N+](C)(C)[O-]", NitrogenChalcogenideKind.AMINE, "N,N-dimethyl-N-oxidomethanaminium"),
        ("C[N+](C)(C)[S-]", NitrogenChalcogenideKind.AMINE, "trimethylazane sulfide"),
        ("C[N+](C)(C)[Se-]", NitrogenChalcogenideKind.AMINE, "trimethylazane selenide"),
        ("C[N+](C)(C)[Te-]", NitrogenChalcogenideKind.AMINE, "trimethylazane telluride"),
        ("CC=[N+](C)[O-]", NitrogenChalcogenideKind.IMINE, "N-methyl-N-oxidoethaniminium"),
        ("CC=[N+](C)[S-]", NitrogenChalcogenideKind.IMINE, "N-methylethanimine N-sulfide"),
        ("CC=[N+](C)[Se-]", NitrogenChalcogenideKind.IMINE, "N-methylethanimine N-selenide"),
        ("CC=[N+](C)[Te-]", NitrogenChalcogenideKind.IMINE, "N-methylethanimine N-telluride"),
    ],
)
def test_nitrogen_chalcogenides_are_classified_before_rendering(smiles: str, kind, expected: str):
    import openclatura as oc

    role = next(role for role in charge_pair_roles(read_smiles(smiles)) if role.nitrogen_kind is not None)

    assert role.nitrogen_kind is kind
    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C#CC[N+](C)([O-])C(C)Cc1ccc(C)cc1", "N-methyl-N-(1-(4-methylphenyl)propan-2-yl)-N-oxidoprop-2-yn-1-aminium"),
        ("C=CC1CC[NH+]([O-])C1", "3-ethenyl-1-oxidopyrrolidinium"),
        ("CC[N+](C)(C)[S-]", "N,N-dimethylethanamine sulfide"),
        ("CC[N+](C)(C)[Se-]", "N,N-dimethylethanamine selenide"),
        ("CC[N+](C)(C)[Te-]", "N,N-dimethylethanamine telluride"),
    ],
)
def test_nitrogen_chalcogenide_citation_route_preserves_whole_parent(smiles: str, expected: str):
    import openclatura as oc

    result = oc.name(smiles, verify_opsin=oc.opsin_available())

    assert result.name == expected
    if oc.opsin_available():
        assert result.opsin_check is not None
        assert result.opsin_check.status == "matched"


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC#[N+][O-]", "ethanenitrile oxide"),
        ("CC#[N+][S-]", "ethanenitrile sulfide"),
        ("CC#[N+][Se-]", "ethanenitrile selenide"),
        ("CC#[N+][Te-]", "ethanenitrile telluride"),
        ("c1ccccc1C#[N+][S-]", "benzenecarbonitrile sulfide"),
    ],
)
def test_nitrile_chalcogenides_use_zwitterion_priority_and_class_names(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("COOC", "1,2-dimethyldioxidane"),
        ("CCOOC", "1-ethyl-2-methyldioxidane"),
        ("CSSC", "1,2-dimethyldisulfane"),
        ("C[Se][Se]C", "1,2-dimethyldiselane"),
        ("C[Te][Te]C", "1,2-dimethylditellane"),
        ("CSOC", "dimethyl thioperoxide"),
        ("CCOSC", "O-ethyl S-methyl thioperoxide"),
        ("CO[Se]C", "dimethyl selenoperoxide"),
        ("CO[Te]C", "dimethyl telluroperoxide"),
        ("CS[Se]C", "dimethyl selenothioperoxide"),
        ("CS[Te]C", "dimethyl tellurothioperoxide"),
        ("C[Se][Te]C", "dimethyl selenotelluroperoxide"),
    ],
)
def test_peroxide_linkages_preserve_both_chalcogen_sites(smiles: str, expected: str):
    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "CC(C)(O)C(C)(C)OSc1ccc(CCCO)cc1",
            "3-(((4-(3-hydroxypropyl)phenyl)sulfanyl)oxy)-2,3-dimethylbutan-2-ol",
        ),
        (
            "C=CC(=O)Oc1ccc(-c2ccc(CCCOSC)cc2)cc1",
            "4-(4-(3-(methylsulfanyloxy)propyl)phenyl)phenyl prop-2-enoate",
        ),
    ],
)
def test_peroxide_linkage_does_not_bypass_a_principal_group(smiles: str, expected: str):
    result = oc.name(smiles, verify_self=True)

    assert result.name == expected
    assert result.self_audit.verdict == "confirmed", result.self_audit.reason
    if oc.opsin_available():
        check = oc.verify_with_opsin(expected, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize(("double_element", "single_element"), product(VALIDATED_CHALCOGENS, repeat=2))
def test_all_acid_site_combinations_have_typed_graph_descriptors(double_element, single_element):
    smiles = f"CC(={SMILES_ATOM[double_element]}){SMILES_HYDROGENATED_ATOM[single_element]}"

    descriptor = _descriptor(smiles, FunctionalFamily.ACYL, DerivativeKind.ACID)

    assert tuple(ligand.element for ligand in descriptor.ligands) == (double_element, single_element)
    assert tuple(ligand.role for ligand in descriptor.ligands) == (
        ChalcogenLigandRole.DOUBLE_BONDED,
        ChalcogenLigandRole.HYDROGEN_BEARING,
    )
    assert oc.name(smiles).name


@pytest.mark.parametrize(("double_element", "single_element"), product(VALIDATED_CHALCOGENS, repeat=2))
def test_all_ester_site_combinations_have_typed_graph_descriptors(double_element, single_element):
    smiles = f"CC(={SMILES_ATOM[double_element]}){SMILES_ATOM[single_element]}C"

    descriptor = _descriptor(smiles, FunctionalFamily.ACYL, DerivativeKind.ESTER)

    assert tuple(ligand.element for ligand in descriptor.ligands) == (double_element, single_element)
    assert tuple(ligand.role for ligand in descriptor.ligands) == (
        ChalcogenLigandRole.DOUBLE_BONDED,
        ChalcogenLigandRole.CARBON_LINK,
    )
    assert oc.name(smiles).name


@pytest.mark.parametrize("double_element", VALIDATED_CHALCOGENS)
@pytest.mark.parametrize(
    ("tail", "derivative"),
    (
        ("N", DerivativeKind.AMIDE),
        ("NN", DerivativeKind.HYDRAZIDE),
        ("Cl", DerivativeKind.ACID_HALIDE),
    ),
)
def test_all_acyl_derivative_elements_share_the_typed_center(double_element, tail, derivative):
    smiles = f"CC(={SMILES_ATOM[double_element]}){tail}"

    descriptor = _descriptor(smiles, FunctionalFamily.ACYL, derivative)

    assert descriptor.ligands[0].element is double_element
    assert descriptor.ligands[0].role is ChalcogenLigandRole.DOUBLE_BONDED
    assert oc.name(smiles).name


@pytest.mark.parametrize("double_element", VALIDATED_CHALCOGENS)
def test_all_urea_chalcogen_elements_share_the_typed_center(double_element):
    smiles = f"NC(={SMILES_ATOM[double_element]})N"

    descriptor = _descriptor(smiles, FunctionalFamily.ACYL, DerivativeKind.UREA)

    assert descriptor.ligands[0].element is double_element
    assert descriptor.ligands[0].role is ChalcogenLigandRole.DOUBLE_BONDED
    assert oc.name(smiles).name


@pytest.mark.parametrize("derivative", (DerivativeKind.ACID, DerivativeKind.ESTER))
@pytest.mark.parametrize(
    ("double_element", "first_linker", "terminal_element"),
    product(VALIDATED_CHALCOGENS, repeat=3),
)
def test_all_peroxy_acyl_site_combinations_have_typed_graph_descriptors(
    derivative,
    double_element,
    first_linker,
    terminal_element,
):
    terminal = (
        SMILES_HYDROGENATED_ATOM[terminal_element]
        if derivative is DerivativeKind.ACID
        else f"{SMILES_ATOM[terminal_element]}C"
    )
    smiles = f"CC(={SMILES_ATOM[double_element]}){SMILES_ATOM[first_linker]}{terminal}"

    descriptor = _descriptor(smiles, FunctionalFamily.ACYL, derivative)

    assert tuple(ligand.element for ligand in descriptor.ligands) == (
        double_element,
        first_linker,
        terminal_element,
    )
    assert tuple(ligand.role for ligand in descriptor.ligands) == (
        ChalcogenLigandRole.DOUBLE_BONDED,
        ChalcogenLigandRole.CHALCOGEN_LINK,
        ChalcogenLigandRole.HYDROGEN_BEARING if derivative is DerivativeKind.ACID else ChalcogenLigandRole.CARBON_LINK,
    )
    assert oc.name(smiles).name


@pytest.mark.parametrize("bridge_length", (1, 2))
def test_all_anhydride_site_combinations_have_typed_graph_descriptors(bridge_length):
    for double_left, double_right, *bridge in product(VALIDATED_CHALCOGENS, repeat=bridge_length + 2):
        bridge_smiles = "".join(SMILES_ATOM[element] for element in bridge)
        smiles = f"CC(={SMILES_ATOM[double_left]}){bridge_smiles}C(C)={SMILES_ATOM[double_right]}"

        descriptor = _descriptor(smiles, FunctionalFamily.ACYL, DerivativeKind.ANHYDRIDE)
        actual_elements = tuple(ligand.element for ligand in descriptor.ligands)
        assert actual_elements == (double_left, double_right, *bridge), (smiles, actual_elements)
        assert oc.name(smiles).name


@pytest.mark.parametrize(
    "smiles",
    (
        "CC(=S)[SeH]",
        "CC(=[Te])[Se]C",
        "CC(=[Se])S[TeH]",
        "CC(=[Te])O[Se]C",
        "CC(=S)[Se][Te]C(C)=O",
    ),
)
def test_mixed_chalcogen_names_are_invariant_to_atom_renumbering(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    baseline = oc.name_mol(mol).name
    atom_count = mol.GetNumAtoms()

    for order in (list(reversed(range(atom_count))), list(range(1, atom_count)) + [0]):
        assert oc.name_mol(Chem.RenumberAtoms(mol, order)).name == baseline


@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        ("CC(=[Se])CC(=O)O", "3-selenoxobutanoic acid"),
        ("CC([TeH])CC(=O)O", "3-tellanylbutanoic acid"),
        ("CC(=[Te])CC(=[Se])C", "4-telluroxopentane-2-selone"),
        ("CC(=[Se])CC(=[Se])C", "pentane-2,4-diselone"),
        ("c1ccccc1C(=[Se])O", "benzenecarboselenoic O-acid"),
        ("CC(=[Se])N(C)C", "N,N-dimethylethaneselenoamide"),
        ("CC(=[Te])N(C)NC", "N,N'-dimethylethanetellurohydrazide"),
        ("[Na+].CC(=[Se])[S-]", "sodium ethaneselenothioate"),
    ),
)
def test_dynamic_chalcogen_rules_compose_with_existing_parent_and_modifier_paths(smiles, expected):
    assert oc.name(smiles).name == expected


def test_polonium_is_parseable_but_has_an_explicit_unvalidated_boundary():
    mol = read_smiles("C[PoH]")

    assert mol.atoms[1].symbol == "Po"
    with pytest.raises(UnsupportedChalcogenNomenclatureError) as exc_info:
        perceive_groups(mol)
    assert str(exc_info.value) == "Organic functional-group nomenclature is not validated for: Po"
