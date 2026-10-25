"""Name-style presets and options (``openclatura.naming_style``)."""

import pytest

import openclatura as oc
from openclatura.naming_style import NamingStyle, nest_enclosing_marks, resolve_style

NESTED = "OC(=O)c1ccccc1OC(C)C(=O)NC(C)C(C)C"


@pytest.mark.parametrize(
    ("smiles", "default", "traditional"),
    [
        ("CC=CC", "but-2-ene", "2-butene"),
        ("CC(C)CC(C)O", "4-methylpentan-2-ol", "4-methyl-2-pentanol"),
        ("OCC=CC", "but-2-en-1-ol", "2-buten-1-ol"),
        ("Oc1ccccc1O", "benzene-1,2-diol", "1,2-benzenediol"),
        ("O=C1CCCC=C1", "cyclohex-2-en-1-one", "2-cyclohexen-1-one"),
        ("C[C@H](O)CC", "(2S)-butan-2-ol", "(2S)-2-butanol"),
        # Non-alphabetic stems and unlocanted parents keep their spelling.
        ("O=C1CC2CCC1C2", "bicyclo[2.2.1]heptan-2-one", "bicyclo[2.2.1]heptan-2-one"),
        ("CCCC(C)C", "2-methylpentane", "2-methylpentane"),
        # Charged parents keep their spelling: the ium locant must not detach.
        ("[O-]C1CCC[NH2+]C1", "piperidinium-3-olate", "piperidinium-3-olate"),
        ("OC(=O)C1CCCC[NH2+]1", "2-carboxypiperidinium", "2-carboxypiperidinium"),
        # Substituent prefixes keep before-suffix locants.
        ("CC(=O)OC(C)CC", "butan-2-yl acetate", "butan-2-yl acetate"),
        # mercapto re-sorts alphanumerically and takes a simple multiplier.
        ("SCCO", "2-sulfanylethan-1-ol", "2-mercapto-1-ethanol"),
        (
            "OC(=O)c1ccc(SC)cc1S",
            "4-(methylsulfanyl)-2-sulfanylbenzoic acid",
            "2-mercapto-4-(methylsulfanyl)benzoic acid",
        ),
        ("OC(=O)C(S)CS", "2,3-bis(sulfanyl)propanoic acid", "2,3-dimercaptopropanoic acid"),
        ("CSCCO", "2-(methylsulfanyl)ethan-1-ol", "2-(methylsulfanyl)-1-ethanol"),
        ("Nc1ccc(S)cc1", "4-aminobenzene-1-thiol", "4-amino-1-benzenethiol"),
    ],
)
def test_traditional_preset(smiles, default, traditional):
    assert oc.name(smiles).name == default
    assert oc.name(smiles, style="traditional").name == traditional


def test_iupac2013_nests_enclosing_marks():
    assert oc.name(NESTED).name == "2-((1-((3-methylbutan-2-yl)amino)-1-oxopropan-2-yl)oxy)benzoic acid"
    assert (
        oc.name(NESTED, style="iupac2013").name == "2-({1-[(3-methylbutan-2-yl)amino]-1-oxopropan-2-yl}oxy)benzoic acid"
    )


def test_alternating_enclosing_marks():
    assert (
        oc.name(NESTED, enclosing_marks="alternating").name
        == "2-[(1-[(3-methylbutan-2-yl)amino]-1-oxopropan-2-yl)oxy]benzoic acid"
    )
    alternating = (("(", ")"), ("[", "]"))
    assert nest_enclosing_marks("a(b(c(d(methyl)ethyl)ethyl)ethyl)", alternating) == "a[b(c[d(methyl)ethyl]ethyl)ethyl]"
    # Descriptor brackets count as level 1, so the mark around them is level 2: parentheses again.
    assert (
        nest_enclosing_marks("4-([1,1'-biphenyl]-4-yl)butanoic acid", alternating)
        == "4-([1,1'-biphenyl]-4-yl)butanoic acid"
    )


@pytest.mark.parametrize(
    ("smiles", "retained", "hantzsch_widman"),
    [
        ("c1ccoc1", "furan", "oxole"),
        ("C1CCNC1", "pyrrolidine", "azolidine"),
        ("c1cncnc1", "pyrimidine", "1,3-diazine"),
        ("C1CCNCC1", "piperidine", "azinane"),
        ("Cc1nc(N)ncc1", "4-methylpyrimidin-2-amine", "4-methyl-1,3-diazin-2-amine"),
        # azine and oxine are never used: pyridine and pyran stay, hydro forms too.
        ("c1ccncc1", "pyridine", "pyridine"),
        ("C1CC=CNC1", "1,2,3,4-tetrahydropyridine", "1,2,3,4-tetrahydropyridine"),
        # Fusion components keep their retained names.
        ("c1ccc2ncccc2c1", "quinoline", "quinoline"),
    ],
)
def test_hantzsch_widman_heterocycle_names(smiles, retained, hantzsch_widman):
    assert oc.name(smiles).name == retained
    assert oc.name(smiles, heterocycle_names="hantzsch_widman").name == hantzsch_widman


def test_override_on_top_of_preset():
    assert oc.name("SCCO", style="iupac2013", thiol_prefix="mercapto").name == "2-mercaptoethan-1-ol"
    assert oc.name("CC=CC", locant_placement="before_parent").name == "2-butene"


def test_style_does_not_leak_between_requests():
    assert oc.name("SCCO", style="traditional").name == "2-mercapto-1-ethanol"
    assert oc.name("SCCO").name == "2-sulfanylethan-1-ol"


def test_batch_and_trace_paths_apply_style():
    results = oc.name_many([NESTED, "SCCO"], style="traditional", processes=2)
    assert [r.name for r in results] == [
        "2-({1-[(3-methylbutan-2-yl)amino]-1-oxopropan-2-yl}oxy)benzoic acid",
        "2-mercapto-1-ethanol",
    ]
    traced = oc.name(NESTED, style="iupac2013", include_trace=True)
    assert traced.analysis.name == traced.name


@pytest.mark.parametrize(
    ("raw", "nested"),
    [
        ("((2R)-butan-2-yl)benzene", "[(2R)-butan-2-yl]benzene"),
        ("(2R,3S)-2-((4-methylphenyl)methyl)butane", "(2R,3S)-2-[(4-methylphenyl)methyl]butane"),
        ("2-(naphthalen-1(2H)-yl)ethanol", "2-[naphthalen-1(2H)-yl]ethanol"),
        ("4-([1,1'-biphenyl]-4-yl)butanoic acid", "4-{[1,1'-biphenyl]-4-yl}butanoic acid"),
        ("bis(2-chloroethyl)amine", "bis(2-chloroethyl)amine"),
        ("(E)-but-2-ene", "(E)-but-2-ene"),
        ("a((b((c((d))))))", "a((b((c((d))))))"),  # no enclosed words: untouched
        ("2-((x)", "2-((x)"),  # unbalanced: untouched
    ],
)
def test_nest_enclosing_marks(raw, nested):
    assert nest_enclosing_marks(raw) == nested
    assert len(nest_enclosing_marks(raw)) == len(raw)


def test_resolve_style_validates():
    assert resolve_style() == NamingStyle()
    assert resolve_style("traditional", thiol_prefix="sulfanyl").thiol_prefix == "sulfanyl"
    with pytest.raises(ValueError, match="unknown naming style"):
        resolve_style("cas")
    with pytest.raises(ValueError, match="unknown thiol_prefix"):
        resolve_style(thiol_prefix="thio")
    with pytest.raises(ValueError, match="unknown style option"):
        resolve_style(brackets="nested")
