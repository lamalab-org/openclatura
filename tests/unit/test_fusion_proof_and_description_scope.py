"""A spiro side keeps retained parents, and proofs survive into the result.

Three defects met at the same boundary: a fused ring system named as part of a
spiro side took a fusion hydride without asking whether a retained parent
outranked it, the proof behind such a name never reached the public result, and
the human description reported the named component as the whole ring system.
"""

import pytest
from rdkit import Chem

from openclatura import name_mol
from openclatura.human_descriptor import describe_human

FUSION_PROOF_CARRIERS = (
    "c1cnc2[nH]cnc2c1",  # the fused system is the parent
    "OC(=O)CCc1nc2ccncc2[nH]1",  # it is a substituent
    "OC(=O)CCOCc1nc2ccncc2[nH]1",  # nested two levels down
    "Cc1ccc2oc(=O)c3c(c2c1)OC(N)=C(C#N)C31C(=O)Nc2ccccc21",  # a spiro side
)


@pytest.mark.parametrize("smiles", FUSION_PROOF_CARRIERS)
def test_a_fusion_proof_reaches_the_public_result(smiles):
    """The engine's proof is reported wherever in the name it was used."""

    result = name_mol(Chem.MolFromSmiles(smiles), include_trace=True)
    assert result.parent_nomenclature == "systematic_fusion", result.name
    assert result.pin_status
    assert result.fusion_support_tier
    assert result.proof_source


@pytest.mark.parametrize("smiles", ("CCCCO", "c1ccc2ccccc2c1", "C1CCCCC1"))
def test_a_name_without_a_fusion_proof_claims_none(smiles):
    """A retained or acyclic parent must not borrow fusion provenance."""

    result = name_mol(Chem.MolFromSmiles(smiles), include_trace=True)
    assert result.parent_nomenclature != "systematic_fusion"


_JOINED = "one component of the fragment's ring system"


def test_the_description_admits_a_spiro_partner_ring():
    """Reporting only the named ring puts the other ring's positions on it."""

    text = "\n".join(describe_human("CCc1ccc(Cc2cc3c(cc2Br)CO[C@]32O[C@H](C)[C@@H](O)[C@H](O)[C@H]2O)s1").paragraphs)
    assert _JOINED in text


@pytest.mark.parametrize(
    "smiles",
    (
        "CCCCO",
        "c1ccc2ccccc2c1",
        "OC(=O)CCc1nc2ccncc2[nH]1",
        "CC(C)(C)OC(=O)N[C@H]1CC[C@H](C(=O)NCc2ccsc2)C1",
        "Cc1cc(Cl)ccc1OCC(=O)Oc1ccc(C(=O)c2ccc(F)cc2)cc1",
    ),
)
def test_an_unjoined_ring_system_is_not_called_a_component(smiles):
    """A ring reached through a plain bond is a substituent, not a partner."""

    text = "\n".join(describe_human(smiles).paragraphs)
    assert _JOINED not in text


def test_a_named_prefix_is_described_by_its_own_name():
    """An acetyl group is not an ethyl group with its carbonyl forgotten."""

    text = "\n".join(describe_human("COC(=O)C[C@]1(C)CC[C@@]2(O1)C(COC(C)=O)=CC[C@H]1C(C)(C)CCC[C@@]12C").paragraphs)
    assert "an acetyl group" in text
    # "ethyl" is the label the two-carbon skeleton alone would earn; guard the
    # standalone word so "methyl" does not satisfy it.
    assert "an ethyl group" not in text
