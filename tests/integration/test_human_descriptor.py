"""Tests for the human-oriented metadata descriptor."""

from __future__ import annotations

from openclatura import HumanDescription, describe_human


def test_human_descriptor_uses_parent_metadata_without_token_spans():
    d = describe_human("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    assert isinstance(d, HumanDescription)
    text = str(d)
    assert "9-membered bicyclic [4.3.0] heteroskeleton" in text
    assert "retained purine parent" in text
    assert "nitrogen at positions 1 (atom id" in text
    assert "3 (atom id" in text
    assert "7 (atom id" in text
    assert "9 (atom id" in text
    assert "oxo groups at positions 2 (atom id" in text
    assert "6 (atom id" in text
    assert "methyl groups at positions 1 (atom id" in text
    assert "token" not in text.lower()
    assert "span" not in text.lower()


def test_human_descriptor_starts_with_processed_smiles_atom_ids():
    d = describe_human("C[C@@H](Cl)C(=O)c1ccccc1")

    first = d.paragraphs[0]
    assert first.startswith("Processed SMILES: C[C@@H](Cl)C(=O)c1ccccc1\n")
    assert "C{0}[C@@H]{1}(Cl{2})C{3}(=O{4})c{5}1c{6}c{7}c{8}c{9}c{10}1" in first


def test_human_descriptor_recurses_into_substituent_parents():
    d = describe_human("CC(=O)Nc1ccccc1")

    text = str(d)
    assert "N-phenylacetamide" in text
    assert "an amide group at position 1 (atom id" in text
    assert "a phenyl group at position N" in text
    assert "phenyl substituent at position N is built around the retained benzene parent" in text
    assert "\nThe principal characteristic feature is an amide group" in text


def test_human_descriptor_uses_local_substituent_names_and_atom_ids():
    d = describe_human("CC(=O)c1cccc(Nc2ccccc2CC)c1")

    text = str(d)
    assert "1-(3-((2-ethylphenyl)amino)phenyl)ethan-1-one" in text
    assert "a phenyl group at position 1 (atom id" in text
    assert "3-((2-ethylphenyl)amino)phenyl group" not in text
    assert "an amino group at position 3 (atom id" in text
    assert "a phenyl group." in text
    assert "a ethyl group" not in text
    assert "an ethyl group" in text


def test_human_descriptor_handles_nested_substituent_trees_generically():
    d = describe_human("O=S(=O)(Nc1nc(cc(n1)C)C)c2ccc(N)cc2")

    text = str(d)
    assert "N-(4-aminophenylsulfonyl)-4,6-dimethylpyrimidin-2-amine" in text
    assert "4-aminophenylsulfonyl" in text
    assert "4-aminophenyl" in text
    assert "retained benzene parent" in text
    assert "amino group" in text


def _body(description) -> list[str]:
    """Description lines, minus the SMILES echo and the "is named" line."""
    return [
        line
        for paragraph in description.paragraphs[1:]
        for line in paragraph.splitlines()
        if not line.startswith("The molecule is named ")
    ]


def test_nested_substituents_are_anchored_to_their_own_subject():
    """Only the principal parent may say "this framework".

    Aspirin's acetoxy branch has no parent metadata of its own, which used to
    drop its subject: its two levels rendered as bare "Attached to this
    framework ..." lines, indistinguishable from further substituents on the
    benzene ring.
    """
    lines = _body(describe_human("CC(=O)Oc1ccccc1C(=O)O"))

    framework = [line for line in lines if "Attached to this framework" in line]
    assert len(framework) == 1, f"only the parent may claim the framework: {lines}"
    assert "an oxy group at position 2" in framework[0]

    # The branch below it names itself instead.
    nested = [line for line in lines if line.startswith("  ")]
    assert nested, f"expected an indented branch: {lines}"
    assert any("The oxy substituent at position 2" in line and "carries" in line for line in nested)


def test_description_indents_by_depth():
    lines = _body(describe_human("CC(=O)c1cccc(Nc2ccccc2CC)c1"))
    depths = {(len(line) - len(line.lstrip(" "))) // 2 for line in lines}

    # phenyl on the parent, amino on it, phenyl on that, ethyl on that.
    assert max(depths) >= 3, f"expected nesting to be indented: {lines}"
    # Indentation is contiguous: no level appears without its parent level.
    assert depths == set(range(max(depths) + 1))


def test_deeper_levels_reuse_the_subject_as_a_pronoun():
    """A node that already named itself says "It carries", not its name twice."""
    lines = _body(describe_human("CC(=O)c1cccc(Nc2ccccc2CC)c1"))

    introduced = [line for line in lines if "is built around" in line and line.startswith(" ")]
    assert introduced, "expected an expanded substituent"
    assert any(line.strip().startswith("It carries ") for line in lines)


def test_reports_atoms_the_naming_metadata_does_not_describe():
    """An ester's alcohol half is absent from the tree; say so, don't drop it."""
    d = describe_human("CC(=O)Oc1ccccc1C(=O)C1CC1")

    coverage = [p for p in d.paragraphs if p.startswith("This description covers")]
    assert coverage, f"expected a coverage note: {d.paragraphs}"
    assert "4 of the 15 atoms" in coverage[0]
    assert "remaining 11" in coverage[0]


def test_no_coverage_note_when_the_whole_structure_is_described():
    for smiles in ("CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"):
        d = describe_human(smiles)
        assert not [p for p in d.paragraphs if p.startswith("This description covers")], smiles
