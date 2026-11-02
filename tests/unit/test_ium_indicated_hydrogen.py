"""A cationic ring parent cites its neutral characteristic groups as prefixes."""

import pytest

import openclatura
from openclatura import name_smiles, opsin_available, verify_with_opsin

# A cation outranks every neutral class (P-41), so a ketone, acid or alcohol on
# the ring that carries the ium centre is a prefix, and the parent's saturation
# is spelled as for its neutral oxo form: indicated hydrogen at the lowest
# saturated locants, hydro prefixes for the rest. P-73.1.2.1 puts the ketones
# of 2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-isoindol-2-ium in prefixes the same
# way; isoindole keeps openclatura's 2H-isoindole spelling, so its name below
# cites the same structure as 1,3-dihydro-2H (lamalab-org/openclatura#134, #163).
CASES = {
    "O=C1C=CC=C[OH+]1": "2-oxo-2H-pyran-1-ium",
    "O=C1C=CC=C[SH+]1": "2-oxo-2H-thiopyran-1-ium",
    "O=C1C=Cc2ccccc2[OH+]1": "2-oxo-2H-1-benzopyran-1-ium",
    "O=C1[OH+]C=Cc2ccccc21": "1-oxo-1H-2-benzopyran-2-ium",
    "O=C1c2ccccc2[OH+]c2ccccc12": "9-oxo-9H-xanthen-10-ium",
    "C[O+]1C=CC(=O)C=C1": "1-methyl-4-oxo-4H-pyran-1-ium",
    "O=C1C=C[NH2+]C=C1": "4-oxo-1,4-dihydropyridin-1-ium",
    "O=C1C=C[NH2+]C=C1O": "3-hydroxy-4-oxo-1,4-dihydropyridin-1-ium",
    "O=C1C=C[NH+](C)C=C1": "1-methyl-4-oxo-1,4-dihydropyridin-1-ium",
    "C[N+]1(C)C=CC(=O)C=C1": "1,1-dimethyl-4-oxo-1,4-dihydropyridin-1-ium",
    "OC(=O)C1=CC(=O)C=C[NH2+]1": "2-carboxy-4-oxo-1,4-dihydropyridin-1-ium",
    "O=C1C=CC=C[NH2+]1": "2-oxo-1,2-dihydropyridin-1-ium",
    "O=C1C=CC=C[NH+]1C": "1-methyl-2-oxo-1,2-dihydropyridin-1-ium",
    "O=C1N=C[NH2+]C=C1": "4-oxo-1,4-dihydropyrimidin-1-ium",
    "O=C1C=C[NH2+]C(=O)N1": "2,4-dioxo-1,2,3,4-tetrahydropyrimidin-1-ium",
    "O=C1NC(=O)C=C[NH+]1C": "1-methyl-2,4-dioxo-1,2,3,4-tetrahydropyrimidin-1-ium",
    "O=C1C=C[NH+]=CC1": "4-oxo-3,4-dihydropyridin-1-ium",
    "O=c1cc[nH]c[nH+]1": "4-oxo-1,4-dihydropyrimidin-3-ium",
    "N=C1C=CNC=[NH+]1": "4-imino-1,4-dihydropyrimidin-3-ium",
    "S=C1C=C[NH2+]C=C1": "4-thioxo-1,4-dihydropyridin-1-ium",
    "C[N+]1(C)C(=O)c2ccccc2C1=O": "2,2-dimethyl-1,3-dioxo-1,3-dihydro-2H-isoindol-2-ium",
    "O=C1C[OH+]C=C1": "3-oxo-2,3-dihydrofuran-1-ium",
    "O=C1CC[NH2+]CC1": "4-oxopiperidinium",
    "OC(=O)c1ccc[nH+]c1": "3-carboxypyridin-1-ium",
    "C[n+]1ccc(C(=O)O)cc1": "4-carboxy-1-methylpyridin-1-ium",
    "C[N+]1=CC=C(C=C1)O": "4-hydroxy-1-methylpyridin-1-ium",
    "OCc1cc[nH+]cc1": "4-(hydroxymethyl)pyridin-1-ium",
}

# TODO(opsin#343): OPSIN 2.9.0 rejects 4-oxo-4H-pyran-1-ium and its thiopyran
# and 1-benzopyran analogues (https://github.com/dan2097/opsin/issues/343), so
# a protonated ring chalcogen opposite the oxo carbon is cited as a hydro
# position. Once a fixed OPSIN is what verify_with_opsin runs, these become
# 4-oxo-4H-pyran-1-ium, 2-carboxy-4-oxo-4H-1-benzopyran-1-ium, and so on.
OPSIN_343_CASES = {
    "O=C1C=C[OH+]C=C1": "4-oxo-1,4-dihydropyran-1-ium",
    "O=C1C=C[OH+]C=C1O": "3-hydroxy-4-oxo-1,4-dihydropyran-1-ium",
    "O=C1C=C[SH+]C=C1": "4-oxo-1,4-dihydrothiopyran-1-ium",
    "O=C1C=C[OH+]c2ccccc12": "4-oxo-1,4-dihydro-1-benzopyran-1-ium",
    "O=C1C=C[SH+]c2ccccc12": "4-oxo-1,4-dihydro-1-benzothiopyran-1-ium",
    "OC(=O)C1=CC(=O)C=C[OH+]1": "2-carboxy-4-oxo-1,4-dihydropyran-1-ium",
    "OC(=O)C1=CC(=O)c2c([OH+]1)cccc2": "2-carboxy-4-oxo-1,4-dihydro-1-benzopyran-1-ium",
}

# Cations with no neutral group on the ring, a group off the ring, a cationic
# suffix, a zwitterion, and the neutral parents keep their names.
GUARDS = {
    "c1cc[nH+]cc1": "pyridin-1-ium",
    "C[n+]1ccccc1": "1-methylpyridin-1-ium",
    "c1c[nH+]c[nH]1": "1H-imidazol-3-ium",
    "c1cc[o+]cc1": "1-oxacyclohexa-1,3,5-trien-1-ium",
    "c1ccc2[o+]cccc2c1": "benzo[b]pyran-1-ium",
    "C1=C[NH2+]C=CC1": "1,4-dihydropyridin-1-ium",
    "OCCc1cc[nH+]cc1": "2-(pyridin-1-ium-4-yl)ethan-1-ol",
    "OC(=O)CC[n+]1ccccc1": "3-(pyridin-1-ium-1-yl)propanoic acid",
    "C[N+](C)(C)CC(=O)O": "1-carboxy-N,N,N-trimethylmethanaminium",
    "[O-][O+]1C=CC(=S)c2ccccc21": "1-oxido-4-thioxo-4H-1-benzopyran-1-ium",
    "O=C1C=COC=C1": "4H-pyran-4-one",
    "O=c1ccoc2ccccc12": "4H-1-benzopyran-4-one",
    "O=C1C=CNC=C1": "pyridin-4(1H)-one",
    "O=C1C=CN(C)C=C1": "1-methylpyridin-4(1H)-one",
}

ALL_NAMES = CASES | OPSIN_343_CASES | GUARDS


@pytest.mark.parametrize("smiles", sorted(CASES | OPSIN_343_CASES))
def test_cationic_ring_cites_neutral_groups_as_prefixes(smiles):
    assert name_smiles(smiles) == (CASES | OPSIN_343_CASES)[smiles]


@pytest.mark.parametrize("smiles", sorted(CASES | OPSIN_343_CASES))
def test_cationic_ring_names_pass_the_final_audit(smiles):
    """The traced path audits the parent's binding term against the final name."""

    result = openclatura.name(smiles, include_trace=True)
    assert result.error is None
    assert result.name == (CASES | OPSIN_343_CASES)[smiles]


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_cationic_ring_guards(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(ALL_NAMES))
def test_cationic_ring_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin(ALL_NAMES[smiles], smiles)
    assert check.status == "matched", check.to_dict()
