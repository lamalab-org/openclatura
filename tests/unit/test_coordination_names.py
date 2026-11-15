"""Splitting additive coordination names that OPSIN cannot read as a whole."""

from openclatura.coordination_names import composition_equivalent, parse_coordination_name

DPPF = "C1(=CC=CC=C1)P([C-]1C=CC=C1)C1=CC=CC=C1.[C-]1(C=CC=C1)P(C1=CC=CC=C1)C1=CC=CC=C1.[Fe+2]"
PPH3 = "c1ccc(cc1)P(c1ccccc1)c1ccccc1"
# PubChem's drawing of (dppf)Cr(CO)4: neutral metals and radical ring carbons.
PUBCHEM_DPPF_CR_CO4 = (
    "[Cr].[Fe].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]."
    "[CH]1[CH][CH][C]([CH]1)P(c2ccccc2)c3ccccc3.[CH]4[CH][CH][C]([CH]4)P(c5ccccc5)c6ccccc6"
)


def fake_opsin(names):
    known = {"1,1'-Bis(diphenylphosphino)ferrocene": DPPF, "triphenylphosphine": PPH3}
    return [known.get(name, "") for name in names]


def test_bracketed_ligand_multiplied_ligand_and_central_atom():
    parsed = parse_coordination_name("[1,1′-Bis(diphenylphosphino)ferrocene]tetracarbonylchromium(0)", opsin=fake_opsin)
    assert parsed.metal == "Cr"
    assert parsed.oxidation_state == 0
    assert parsed.ligands == (("1,1'-Bis(diphenylphosphino)ferrocene", 1), ("carbonyl", 4))
    assert parsed.smiles == ".".join(["[Cr]", DPPF, *["[C-]#[O+]"] * 4])
    assert composition_equivalent(parsed.smiles, PUBCHEM_DPPF_CR_CO4)


def test_a_different_count_or_metal_does_not_match():
    for name in (
        "[1,1'-Bis(diphenylphosphino)ferrocene]tricarbonylchromium(0)",
        "[1,1'-Bis(diphenylphosphino)ferrocene]tetracarbonylmolybdenum(0)",
    ):
        parsed = parse_coordination_name(name, opsin=fake_opsin)
        assert not composition_equivalent(parsed.smiles, PUBCHEM_DPPF_CR_CO4)


def test_oxidation_number_and_ligand_charges_give_the_complex_charge():
    neutral = parse_coordination_name("dichlorobis(triphenylphosphine)palladium(II)", opsin=fake_opsin)
    assert neutral.smiles.startswith("[Pd+2].[Cl-].[Cl-].")
    assert neutral.charge == 0
    cation = parse_coordination_name("tetraamminedichloridocobalt(III)", opsin=fake_opsin)
    assert cation.smiles == "[Co+3].N.N.N.N.[Cl-].[Cl-]"
    assert cation.charge == 1


def test_binding_descriptors_are_ignored():
    parsed = parse_coordination_name("bis(η5-cyclopentadienyl)iron(II)", opsin=fake_opsin)
    assert parsed.smiles == "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1"


def test_names_that_are_not_coordination_names_are_left_alone():
    assert parse_coordination_name("benzene", opsin=fake_opsin) is None
    assert parse_coordination_name("sodium chloride", opsin=fake_opsin) is None
    assert parse_coordination_name("bis(unknownligand)copper(II)", opsin=fake_opsin) is None
