"""Coordination compounds: every input drawing reduces to metal + oxidation state + ligands."""

from pathlib import Path

import pytest

from openclatura import NamingEngine, NamingRequest, name_molblock, name_smiles
from openclatura.coordination import perceive

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "coordination"

# PubChem's drawing of (dppf)Cr(CO)4: neutral metals, radical ring carbons, no bonds to either metal.
PUBCHEM_DPPF_CR_CO4 = (
    "[Cr].[Fe].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]."
    "[CH]1[CH][CH][C]([CH]1)P(c2ccccc2)c3ccccc3.[CH]4[CH][CH][C]([CH]4)P(c5ccccc5)c6ccccc6"
)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (PUBCHEM_DPPF_CR_CO4, "[1,1'-bis(diphenylphosphanyl)ferrocene]tetracarbonylchromium(0)"),
        ("[Cr].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]", "hexacarbonylchromium(0)"),
        # However the carbonyls are drawn, they are neutral CO and the metal stays at zero.
        ("O=C=[Fe](=C=O)(=C=O)(=C=O)=C=O", "pentacarbonyliron(0)"),
        # An ordinary bond gives its pair to the ligand; a dative bond leaves the ligand neutral.
        ("N->[Pt](<-N)(Cl)Cl", "diamminedichloridoplatinum(II)"),
        ("Cl[Ti](Cl)(Cl)Cl", "tetrachloridotitanium(IV)"),
        (
            "Cl[Pd](Cl)(<-P(c1ccccc1)(c1ccccc1)c1ccccc1)<-P(c1ccccc1)(c1ccccc1)c1ccccc1",
            "dichloridobis(triphenylphosphane-κP)palladium(II)",
        ),
        ("CC(=O)O[Cu]OC(C)=O", "bis(acetato-κO)copper(II)"),
        # Components not bonded to the centre stay outside the coordination sphere.
        ("N->[Co+3](<-N)(<-N)(<-N)(<-N)<-N.[Cl-].[Cl-].[Cl-]", "hexaamminecobalt(III) trichloride"),
        # Anionic complexes take the -ate name and follow their cations.
        ("[K+].[K+].Cl[Pt-2](Cl)(Cl)Cl", "dipotassium tetrachloridoplatinate(II)"),
        ("N#C[Fe-4](C#N)(C#N)(C#N)(C#N)C#N.[K+].[K+].[K+].[K+]", "tetrapotassium hexacyanidoferrate(II)"),
        # One ring on a metal is a ligand, not a metallocene.
        (
            "[Mn].[CH]1[CH][CH][CH][CH]1.[C-]#[O+].[C-]#[O+].[C-]#[O+]",
            "tricarbonyl(η5-cyclopentadienyl)manganese(I)",
        ),
    ],
)
def test_complex_names(smiles, expected):
    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(
    "smiles",
    [
        "[Fe].[CH]1[CH][CH][CH][CH]1.[CH]1[CH][CH][CH][CH]1",  # PubChem: radical rings, neutral metal
        "C12C3C4C5C1[Fe]23451234C5C1C2C3C45",  # ten Fe-C bonds
        "[cH-]1cccc1.[cH-]1cccc1.[Fe+2]",  # ionic
    ],
)
def test_every_ferrocene_drawing_is_ferrocene(smiles):
    assert name_smiles(smiles) == "ferrocene"
    assert perceive(smiles).complex is None


def test_substituted_ferrocene_in_the_radical_drawing():
    assert name_smiles("CC(=O)[C]1[CH][CH][CH][CH]1.[CH]1[CH][CH][CH][CH]1.[Fe]") == "acetylferrocene"


def test_the_model_records_metal_oxidation_state_ligands_and_donors():
    complex_ = perceive("N->[Pt](<-N)(Cl)Cl").complex
    assert (complex_.metal, complex_.oxidation_state, complex_.charge) == ("Pt", 2, 0)
    assert {(ligand.smiles, ligand.count, ligand.charge, ligand.donors) for ligand in complex_.ligands} == {
        ("N", 2, 0, ("N",)),
        ("[Cl-]", 2, -1, ("Cl",)),
    }


def test_molfile_input_keeps_dative_and_haptic_bonds():
    dppf = name_molblock((EXAMPLES / "dppf_cr_co4.mol").read_text())
    assert dppf.name == "[1,1'-bis(diphenylphosphanyl)ferrocene-κ2P,P']tetracarbonylchromium(0)"
    assert name_molblock((EXAMPLES / "ferrocene_haptic.mol").read_text()).name == "ferrocene"


@pytest.mark.parametrize(
    "smiles",
    [
        "[Fe+2].[Cl-].[Cl-]",  # a salt
        "[Cr].[Mo].[C-]#[O+]",  # two centres
        "[Cr][Cr]",  # metal-metal bond
    ],
)
def test_unsupported_metal_inputs_fail_loudly(smiles):
    result = NamingEngine().run(NamingRequest(smiles=smiles))
    assert result.name == ""
    assert "UnsupportedMetalError" in result.error


def test_inputs_without_a_transition_metal_are_untouched():
    assert name_smiles("CCO") == "ethanol"
    assert name_smiles("C1CC[Mg]C1") == "1-magnesacyclopentane"


# Pd(dppf)Cl2 in the ways it actually gets drawn.
PD_DPPF_CL2 = "dichloridopalladium(II)"


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # PubChem: Cl-Pd-Cl drawn, the phosphane and iron left beside it.
        (
            "Cl[Pd]Cl.[Fe].[CH]1[CH][CH][C]([CH]1)P(c2ccccc2)c3ccccc3.[CH]4[CH][CH][C]([CH]4)P(c5ccccc5)c6ccccc6",
            "[1,1'-bis(diphenylphosphanyl)ferrocene]" + PD_DPPF_CL2,
        ),
        # Dative P->Pd.
        (
            "Cl[Pd](Cl)(<-P(c1ccccc1)(c1ccccc1)[c-]1cccc1)<-P(c1ccccc1)(c1ccccc1)[c-]1cccc1.[Fe+2]",
            "[1,1'-bis(diphenylphosphanyl)ferrocene-κ2P,P']" + PD_DPPF_CL2,
        ),
        # As in a structure diagram: plain lines P-Pd and ten Fe-C bonds.
        (
            "Cl[Pd]1(Cl)P(c2ccccc2)(c2ccccc2)C23C4C5C6C2[Fe]34567892C3C7C8C9C32P1(c1ccccc1)c1ccccc1",
            "[1,1'-bis(diphenylphosphanyl)ferrocene-κ2P,P']" + PD_DPPF_CL2,
        ),
    ],
)
def test_pd_dppf_dichloride_from_each_drawing(smiles, expected):
    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # A plain line from a donor already at its normal valence is a lone-pair donation.
        ("CS(C)[Pt](Cl)(Cl)S(C)C", "dichloridobis(dimethylsulfane-κS)platinum(II)"),
        ("Cl[Pt](Cl)(n1ccccc1)n1ccccc1", "dichloridobis(pyridine-κN)platinum(II)"),
        # Hydrogens on the metal are hydrido ligands, not dropped.
        ("[H][Co](C#O)(C#O)(C#O)C#O", "tetracarbonylhydridocobalt(I)"),
    ],
)
def test_plain_bond_donors_and_metal_hydrides(smiles, expected):
    assert name_smiles(smiles) == expected
