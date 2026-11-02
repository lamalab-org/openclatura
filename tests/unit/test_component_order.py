"""A salt's anion directly follows its cations; neutral components come after the salt."""

import pytest

from openclatura import name_smiles, opsin_available, verify_with_opsin

# Cited between the cation and the anion, an alcohol or phenol name is read with
# the anion as one functional class name: ``propan-2-ol methanesulfonate`` is the
# ester propan-2-yl methanesulfonate (lamalab-org/openclatura#135).
CASES = {
    "OC(C)C.CS(=O)(=O)[O-].C[N+](C)(C)C": "N,N,N-trimethylmethanaminium methanesulfonate propan-2-ol",
    "CC(O)C.Cc1ccc(cc1)S(=O)(=O)[O-].C[N+](C)(C)C": (
        "N,N,N-trimethylmethanaminium 4-methylbenzene-1-sulfonate propan-2-ol"
    ),
    "OCC.CC[N+](CC)(CC)CC.CS(=O)(=O)[O-]": "N,N,N-triethylethanaminium methanesulfonate ethanol",
    "CCO.[NH4+].CS(=O)(=O)[O-]": "ammonium methanesulfonate ethanol",
    "CCO.[Na+].CC(=O)[O-]": "sodium acetate ethanol",
    "Oc1ccccc1.[Na+].CC(=O)[O-]": "sodium acetate phenol",
    "CCO.[Na+].[Na+].[O-]S(=O)(=O)[O-]": "disodium sulfate ethanol",
    "CCO.CC(=O)[O-].CC(=O)[O-].[Ca+2]": "calcium diacetate ethanol",
    "CCO.C[N+](C)(C)C.[Cl-]": "N,N,N-trimethylmethanaminium chloride ethanol",
    "[Na+].[Cl-].O": "sodium chloride water",
    "C1CCOC1.[Li+].[Cl-]": "lithium chloride oxolane",
    "CC(=O)O.C[N+](C)(C)C.[Cl-]": "N,N,N-trimethylmethanaminium chloride acetic acid",
    "OCCO.[K+].[O-]C(=O)c1ccccc1": "potassium benzoate ethane-1,2-diol",
    "[Na+].[Cl-].Cl": "sodium chloride hydrogen chloride",
}

# Salts without a neutral component, and neutral components without a salt, keep
# their order.
GUARDS = {
    "OCC.CC(=O)O": "acetic acid ethanol",
    "CCN.Cl": "ethanamine hydrogen chloride",
    "[NH4+].[NH4+].[O-]S([O-])(=O)=O": "diammonium sulfate",
    "[Na+].[K+].[O-]S([O-])(=O)=O": "potassium sodium sulfate",
    "CC(=O)[O-].[Na+]": "sodium acetate",
    "C[N+](C)(C)C.[Cl-].[Br-].C[N+](C)(C)C": "bis(N,N,N-trimethylmethanaminium) bromide chloride",
}


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_neutral_components_follow_the_salt(smiles):
    assert name_smiles(smiles) == CASES[smiles]


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_component_order_guards(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES | GUARDS))
def test_component_order_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin((CASES | GUARDS)[smiles], smiles)
    assert check.status == "matched", check.to_dict()
