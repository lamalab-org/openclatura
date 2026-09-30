"""Ranked fusion-name candidates stop at the configured budget.

Preference-ordered candidates prove a supported decomposition almost at once
or never, so the walk past that point only enumerates ways to fail, each one
paying for a full numbering and reconstruction audit. Bounding the walk must
not cost a name that the unbounded walk would have produced.
"""

import pytest
from rdkit import Chem

from openclatura import name_mol, name_smiles
from openclatura.chains import find_ring_systems
from openclatura.fusion import planner
from openclatura.fusion.config import fusion_nomenclature_config
from openclatura.fusion.model import FusionConfirmed, FusionUnsupported
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles

# Ring systems the planner names itself, rather than ones carrying a retained
# name, so the budget is exercised on the path it actually bounds.
PLANNED = (
    ("c1cnc2[nH]cnc2c1", "3H-imidazo[4,5-b]pyridine"),
    ("Brc1cn2c(n1)OCCC2", "2-bromo-6,7-dihydro-5H-imidazo[2,1-b][1,3]oxazine"),
    ("CSc1nc(N)c2ccc3ccccc3c2n1", "2-(methylsulfanyl)naphtho[1,2-d]pyrimidin-4-amine"),
    ("Cc1nc2cc(O)c3ccccc3c2o1", "2-methylnaphtho[2,1-d][1,3]oxazol-5-ol"),
    ("Cc1c2ccccc2nc2cc3ccccc3n12", "12-methylindolo[1,2-a]benzo[d]pyrimidine"),
    ("COC(=O)c1cc2c(cn1)ncn2Cc1ccc(F)cc1", "methyl 1-((4-fluorophenyl)methyl)imidazo[4,5-c]pyridine-6-carboxylate"),
)


def _largest_ring_system(smiles):
    mol = read_smiles(smiles)
    return mol, max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms


def test_budget_is_configured_and_positive():
    limit = fusion_nomenclature_config().search.maximum_name_candidates
    assert limit >= 1
    assert planner.MAXIMUM_NAME_CANDIDATES == limit


@pytest.mark.parametrize(("smiles", "expected"), PLANNED)
def test_the_planner_owns_these_names(smiles, expected):
    named = name_mol(Chem.MolFromSmiles(smiles), include_trace=True)
    assert named.error is None
    assert named.name == expected
    assert named.parent_nomenclature == "systematic_fusion"


@pytest.mark.parametrize(("smiles", "expected"), PLANNED)
def test_the_configured_budget_costs_no_proven_name(smiles, expected):
    assert name_smiles(smiles) == expected
    mol, atoms = _largest_ring_system(smiles)
    assert isinstance(plan_fusion_parent(mol, atoms, mode="audited_pin"), FusionConfirmed)


@pytest.mark.parametrize(("smiles", "expected"), PLANNED)
def test_one_candidate_already_proves_the_preferred_decomposition(monkeypatch, smiles, expected):
    """Candidates are emitted in preference order, so the first one carries it."""

    monkeypatch.setattr(planner, "MAXIMUM_NAME_CANDIDATES", 1)
    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(("smiles", "expected"), PLANNED)
def test_an_exhausted_budget_abstains_with_a_typed_reason(monkeypatch, smiles, expected):
    """Refusing every candidate abstains; it never raises and never guesses."""

    monkeypatch.setattr(planner, "MAXIMUM_NAME_CANDIDATES", 0)
    mol, atoms = _largest_ring_system(smiles)
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionUnsupported), result
    assert "candidate budget" in result.reason


@pytest.mark.parametrize(("smiles", "expected"), PLANNED)
def test_an_exhausted_budget_still_yields_a_name(monkeypatch, smiles, expected):
    """A bounded planner falls back rather than leaving the caller nameless."""

    monkeypatch.setattr(planner, "MAXIMUM_NAME_CANDIDATES", 0)
    named = name_mol(Chem.MolFromSmiles(smiles), include_trace=True)
    assert named.name
    assert named.parent_nomenclature != "systematic_fusion"


def test_a_deeply_ambiguous_ring_system_stays_within_the_budget():
    """The system that motivated the budget once walked 294 candidates."""

    smiles = "CC(C)=CCC[C@]1(C)Oc2cc(CO)c(C=O)c3c2[C@H]2[C@@H](O3)[C@@](C)(O)CC[C@H]21"
    counted = {"calls": 0}
    original = planner._plan_numbered_candidate

    def counting(*args, **kwargs):
        counted["calls"] += 1
        return original(*args, **kwargs)

    planner._plan_numbered_candidate = counting
    try:
        assert name_smiles(smiles)
    finally:
        planner._plan_numbered_candidate = original
    assert 0 < counted["calls"] <= 2 * planner.MAXIMUM_NAME_CANDIDATES
