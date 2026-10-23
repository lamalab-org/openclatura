"""Fusion candidates are bounded by exact parent coverage, not a count cap."""

import pytest
from rdkit import Chem

from openclatura import name_mol, name_smiles
from openclatura.chains import find_ring_systems
from openclatura.fusion import planner
from openclatura.fusion.descriptor import FusionDescriptorError, iter_fusion_name_asts
from openclatura.fusion.faces import cached_bounded_face_model
from openclatura.fusion.model import FusionConfirmed, FusionMode, FusionUnsupported
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_smiles

# Ring systems the planner names itself, rather than ones carrying a retained
# name, so they exercise the ranked systematic-fusion path.
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


def _parent_edges(mol, atoms):
    return frozenset(
        tuple(sorted((bond.u, bond.v))) for bond in mol.bonds.values() if bond.u in atoms and bond.v in atoms
    )


@pytest.mark.parametrize(("smiles", "expected"), PLANNED)
def test_the_planner_owns_these_names(smiles, expected):
    named = name_mol(Chem.MolFromSmiles(smiles), include_trace=True)
    assert named.error is None
    assert named.name == expected
    assert named.parent_nomenclature == "systematic_fusion"


@pytest.mark.parametrize(("smiles", "expected"), PLANNED)
def test_exact_parent_coverage_costs_no_proven_name(smiles, expected):
    assert name_smiles(smiles) == expected
    mol, atoms = _largest_ring_system(smiles)
    assert isinstance(plan_fusion_parent(mol, atoms, mode="audited_pin"), FusionConfirmed)


def test_candidate_iteration_has_no_arbitrary_count_limit(monkeypatch):
    mol, atoms = _largest_ring_system(PLANNED[0][0])
    confirmed = planner._plan_uncached(mol, atoms, FusionMode.AUDITED_PIN)
    assert isinstance(confirmed, FusionConfirmed)
    candidates = tuple(object() for _ in range(80))
    attempted = []

    def generated(*args, **kwargs):
        yield from candidates

    def complete(mol, atoms, mode, ast, *args, **kwargs):
        attempted.append(ast)
        return confirmed if ast is candidates[-1] else FusionUnsupported("not this construction")

    monkeypatch.setattr(planner, "iter_fusion_name_asts", generated)
    monkeypatch.setattr(planner, "_plan_numbered_candidate", complete)

    assert planner._plan_uncached(mol, atoms, FusionMode.AUDITED_PIN) is confirmed
    assert attempted == list(candidates)


def test_incomplete_match_union_is_rejected_before_cover_enumeration(monkeypatch):
    mol, atoms = _largest_ring_system("c1ccc2ccccc2c1")
    bounded = cached_bounded_face_model(mol, atoms)
    registry = fusion_component_registry()
    matches = registry.match_faces(mol, bounded)
    assert matches
    unavailable_atom = max(mol.atoms) + 1

    def must_not_enumerate(*args, **kwargs):
        pytest.fail("an incomplete global match union must be rejected before cover enumeration")

    monkeypatch.setattr("openclatura.fusion.descriptor._exact_component_covers", must_not_enumerate)
    with pytest.raises(FusionDescriptorError, match="do not cover the selected parent atom set"):
        next(
            iter_fusion_name_asts(
                mol,
                matches,
                registry,
                required_parent_atom_ids=atoms | {unavailable_atom},
                required_parent_edges=_parent_edges(mol, atoms),
            )
        )


def test_component_cover_must_reconstruct_the_required_parent_edges():
    mol, atoms = _largest_ring_system("c1ccc2ccccc2c1")
    bounded = cached_bounded_face_model(mol, atoms)
    registry = fusion_component_registry()
    matches = registry.match_faces(mol, bounded)
    edges = _parent_edges(mol, atoms)

    with pytest.raises(FusionDescriptorError, match="no exact tree-cover fusion citation"):
        next(
            iter_fusion_name_asts(
                mol,
                matches,
                registry,
                required_parent_atom_ids=atoms,
                required_parent_edges=frozenset(tuple(edges)[1:]),
            )
        )


def test_deeply_ambiguous_system_prunes_incomplete_parents_before_candidate_audit(monkeypatch):
    smiles = "C=C1/C=C\\C=C/N(c2ccc(C)cc2)c2c3c(c4c(c21)-c1ccccc1C4)C=CC3"
    attempted = 0
    original = planner._plan_numbered_candidate

    def counting(*args, **kwargs):
        nonlocal attempted
        attempted += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, "_plan_numbered_candidate", counting)

    assert name_smiles(smiles)
    assert attempted == 4
