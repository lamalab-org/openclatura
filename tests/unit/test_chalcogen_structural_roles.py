"""Structural tests for composable chalcogen roles."""

import pytest

import openclatura as oc
from openclatura.chalcogen_roles import AcylLeavingGroup, Chalcogen, DerivativeKind, FunctionalFamily
from openclatura.graph_io import read_smiles
from openclatura.perception import perceive_groups


def _groups_in_family(smiles: str, family: FunctionalFamily):
    return [
        group
        for group in perceive_groups(read_smiles(smiles))
        if group.descriptor is not None and group.descriptor.family is family
    ]


def test_repeated_chalcogen_linkages_remain_distinct_perceived_groups():
    linkages = _groups_in_family("CC(OSC)(O[Se]C)C", FunctionalFamily.PEROXIDE)

    assert len(linkages) == 2
    assert len({group.descriptor.centers for group in linkages}) == 2


def test_repeated_nitrogen_chalcogenides_remain_distinct_perceived_groups():
    linkages = _groups_in_family(
        "[O-][N+](C)(C)CC[N+](C)(C)[S-]",
        FunctionalFamily.NITROGEN_CHALCOGENIDE,
    )

    assert len(linkages) == 2
    assert len({group.descriptor.centers for group in linkages}) == 2


@pytest.mark.parametrize(
    ("smiles", "kind", "unit_size"),
    [
        ("CC(=O)N=[N+]=[N-]", AcylLeavingGroup.AZIDE, 3),
        ("CC(=O)C#N", AcylLeavingGroup.CYANIDE, 2),
        ("CC(=O)N#C", AcylLeavingGroup.ISOCYANIDE, 2),
        ("CC(=O)N=C=O", AcylLeavingGroup.ISOCYANATE, 3),
        ("CC(=O)N=C=S", AcylLeavingGroup.ISOTHIOCYANATE, 3),
        ("CC(=O)N=C=[Se]", AcylLeavingGroup.ISOSELENOCYANATE, 3),
        ("CC(=O)N=C=[Te]", AcylLeavingGroup.ISOTELLUROCYANATE, 3),
    ],
)
def test_acyl_pseudohalide_owns_complete_leaving_unit(smiles: str, kind: AcylLeavingGroup, unit_size: int):
    matches = [
        group
        for group in perceive_groups(read_smiles(smiles))
        if group.descriptor is not None and group.descriptor.derivative is DerivativeKind.ACYL_PSEUDOHALIDE
    ]

    assert len(matches) == 1
    descriptor = matches[0].descriptor
    assert descriptor.leaving_group is not None
    assert descriptor.leaving_group.kind is kind
    assert len(descriptor.leaving_group.atom_ids) == unit_size
    assert len(descriptor.leaving_group.bond_ids) == unit_size
    assert set(descriptor.leaving_group.atom_ids) <= matches[0].atoms_involved
    assert set(descriptor.leaving_group.bond_ids) <= matches[0].bond_ids


@pytest.mark.parametrize(
    ("smiles", "elements"),
    [
        ("N#CC1CC(=O)NC1=O", {Chalcogen.OXYGEN}),
        ("N#CC1CC(=[Se])NC1=[Te]", {Chalcogen.SELENIUM, Chalcogen.TELLURIUM}),
    ],
)
def test_eligible_cyclic_imide_owns_both_acyl_centers(smiles: str, elements: set[Chalcogen]):
    matches = [
        group
        for group in perceive_groups(read_smiles(smiles))
        if group.descriptor is not None and group.descriptor.derivative is DerivativeKind.IMIDE
    ]

    assert len(matches) == 1
    group = matches[0]
    descriptor = group.descriptor
    assert len(descriptor.centers) == 2
    assert len(descriptor.shared_atoms) == 1
    assert descriptor.linker_paths == ((descriptor.centers[0], descriptor.shared_atoms[0], descriptor.centers[1]),)
    assert {ligand.element for ligand in descriptor.ligands} == elements
    assert descriptor.atom_ids <= group.atom_ids
    assert {ligand.bond_id for ligand in descriptor.ligands} <= group.bond_ids


def test_acyclic_shared_nitrogen_is_not_promoted_to_retained_imide():
    matches = [
        group
        for group in perceive_groups(read_smiles("CC(=O)NC(C)=O"))
        if group.descriptor is not None and group.descriptor.derivative is DerivativeKind.IMIDE
    ]

    assert matches == []


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("N#CC1CC(=O)NC1=O", "3-cyanopyrrolidine-2,5-dione"),
        ("N#CC1CC(=[Se])NC1=[Te]", "4-cyano-5-telluroxopyrrolidine-2-selone"),
    ],
)
def test_cyclic_imide_route_outranks_nitrile_without_replacing_member_suffixes(smiles: str, expected: str):
    assert oc.name(smiles).name == expected


def test_cyclic_imide_route_remains_below_a_carboxylic_ester():
    smiles = "CCOC(=O)C12CC(C1)[S@](=O)(=NC(=O)CN1C(=O)NC(C)(C)C1=O)C2"
    assert oc.name(smiles).name == (
        "ethyl (2S)-2-((2-(4,4-dimethyl-2,5-dioxoimidazolidin-1-yl)acetyl)imino)-2-oxo-"
        "2lambda^6-thiabicyclo[2.1.1]hexane-4-carboxylate"
    )


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=O)N=[N+]=[N-]", "acetyl azide"),
        ("CCCC(=O)C#N", "butanoyl cyanide"),
        ("CC(=O)[N+]#[C-]", "acetyl isocyanide"),
        ("CC(=O)N=C=O", "acetyl isocyanate"),
        ("CC(=O)N=C=S", "acetyl isothiocyanate"),
        ("CC(=O)N=C=[Se]", "acetyl isoselenocyanate"),
        ("CC(=O)N=C=[Te]", "acetyl isotellurocyanate"),
        ("CC(=S)N=C=O", "ethanethioyl isocyanate"),
        ("CC(=[Se])N=C=S", "ethaneselenoyl isothiocyanate"),
    ],
)
def test_acyl_pseudohalides_use_the_complete_leaving_group_route(smiles: str, expected: str):
    assert oc.name(smiles).name == expected


def test_acyl_pseudohalide_priority_composes_with_an_amide():
    assert oc.name("NC(=O)CCC(=O)N=C=O").name == "3-carbamoylpropanoyl isocyanate"
