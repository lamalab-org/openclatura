from openclatura import name


def test_charged_ring_is_parent_instead_of_an_ideyl_substituent():
    smiles = "C1=CC=C([C-]=C1)C2=CC=CC=N2"

    result = name(smiles, verify_opsin=True)

    assert result.error is None
    assert result.name == "2-(pyridin-2-yl)benzen-1-ide"
    assert "ideyl" not in result.name
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched"


def test_charged_ring_substituent_keeps_charge_and_attachment_suffixes_separate():
    smiles = "[C-]1=CC=CC=C1C2=CC=C(C=C2)C3=CC=C(C=C3)O"

    result = name(smiles, verify_opsin=True)

    assert result.error is None
    assert "ideyl" not in result.name
    assert "benzen-1-ide-2-yl" in result.name or "benzen-2-ide-1-yl" in result.name
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched"


def test_one_carbon_charged_substituent_uses_separate_suffixes():
    smiles = "[CH-](O)c1ccc(C(=O)O)cc1"

    result = name(smiles, verify_opsin=True)

    assert result.error is None
    assert result.name == "4-(hydroxymethan-1-ide-1-yl)benzoic acid"
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched"


def test_structured_anion_suffix_is_not_repeated_by_legacy_rewrite():
    smiles = "CCCC1=C(C2=NC3=NC(=NC4=C(C(=C([N-]4)N=C5C(=C(C(=N5)N=C1[N-]2)CCC)CCC)CCC)CCC)C(=C3N(C)C)N(C)C)CCC.[Zn+2]"

    result = name(smiles)

    assert result.error is None
    assert "-21,23-diide-4,5-diamine" in result.name
    assert "-21-ide-23-ide" not in result.name


def test_positive_amide_nitrogen_uses_amidium_suffix():
    smiles = "CC(=O)[N+](C)(CC1=CC=CC=C1)CC(=O)C2=CC=CC=C2.[Cl-]"

    result = name(smiles, verify_opsin=True)

    assert result.error is None
    assert result.name == "N-benzyl-N-methyl-N-(2-oxo-2-phenylethyl)acetamidium chloride"
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched"


def test_retained_and_systematic_ring_amides_share_cationic_suffix_rule():
    cases = {
        "C[N+](C)(C)C(=O)c1ccccc1.[Cl-]": "N,N,N-trimethylbenzamidium chloride",
        "C[N+](C)(C)C(=O)C1CCCCC1.[Cl-]": "N,N,N-trimethylcyclohexanecarboxamidium chloride",
    }

    for smiles, expected in cases.items():
        result = name(smiles, verify_opsin=True)
        assert result.error is None
        assert result.name == expected
        assert result.opsin_check is not None
        assert result.opsin_check.status == "matched"
