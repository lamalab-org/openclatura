"""The mancude reference, and the hydrogen it owes, are arithmetic.

P-25.7.1.1 fixes a fused parent as the maximum number of noncumulative double
bonds the COMPLETED system admits. Indicated hydrogen then distinguishes
hydrogen-placement isomers *within* that maximum family (P-14.7.1 prints both
1H- and 3H-pyrrole); it never authorises a smaller maximum. Two counts follow
and neither is a preference:

    u       = n_pi - 2*D0            indicated-H sites the reference needs
    q_hydro = 2*(D0 - D_target)      hydrogen the hydro prefix must supply

where n_pi counts skeletal sites able to take one noncumulative double bond,
so a neutral nitrogen already holding three skeletal bonds is excluded - it
cannot pair and carries no hydrogen either.

So indicated-H count and hydro count are not competing objectives. P-31.2.2
gives indicated hydrogen priority over hydro *for low locants*, not as an
interchangeable way of supplying hydrogen: for one reference the hydro
multiplier is fixed, and only the locants can move. A candidate offering a
different hydro total for the same reference has failed feasibility, not lost
a preference.
"""

import pytest
from rdkit import Chem

from openclatura import name_smiles, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionMode
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import _abstract_graph, _plan_numbered_candidate, plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol

HYDROGENATED_CARBAZOLE = "c1ccc2c3c([nH]c2c1)CCCC3"
INDOLOQUINOLIZINE = "C1=CC=CN2CCC3C4=CC=CC=C4NC3C12"


def _reference_counts(smiles, monkeypatch):
    """Return (D0, n_pi, target double bonds) for the first planned candidate."""

    mol = read_rdkit_mol(Chem.MolFromSmiles(smiles))
    core = max(find_ring_systems(mol), key=lambda system: len(system.atoms))
    captured = {}
    original = _plan_numbered_candidate

    def capture(graph_mol, atoms, mode, ast, registry, *args, **kwargs):
        captured.setdefault("ast", (ast, registry))
        return original(graph_mol, atoms, mode, ast, registry, *args, **kwargs)

    monkeypatch.setattr("openclatura.fusion.planner._plan_numbered_candidate", capture)
    plan_fusion_parent(mol, core.atoms, mode=FusionMode.GENERAL)
    graph = _abstract_graph(*captured["ast"])

    degree = {atom.id: sum(1 for bond in graph.bonds if atom.id in bond.atoms) for atom in graph.atoms}
    n_pi = sum(1 for atom in graph.atoms if not (atom.symbol == "N" and degree[atom.id] >= 3))
    target_double = sum(
        1 for bond in mol.bonds.values() if bond.u in core.atoms and bond.v in core.atoms and bond.order == 2
    )
    return parent_bond_model(graph).maximum_non_cumulative_double_bonds, n_pi, target_double


@pytest.mark.parametrize(
    "smiles,expected_maximum,expected_sites,expected_hydro",
    (
        pytest.param(HYDROGENATED_CARBAZOLE, 6, 1, 4, id="carbazole-one-site-four-hydro"),
        pytest.param(INDOLOQUINOLIZINE, 8, 0, 6, id="indoloquinolizine-no-site-six-hydro"),
    ),
)
def test_the_reference_fixes_both_counts(smiles, expected_maximum, expected_sites, expected_hydro, monkeypatch):
    maximum, n_pi, target_double = _reference_counts(smiles, monkeypatch)

    assert maximum == expected_maximum
    assert n_pi - 2 * maximum == expected_sites
    assert 2 * (maximum - target_double) == expected_hydro


# KNOWN DEFECT. The reference above is right - the engine reports the maximum
# as eight - but the derivative proof then returns four hydrogenated atoms for
# a target six short of it, and reports the state compatible. Four cannot be
# right against an eight-double-bond reference and a five-double-bond target,
# because 2*(8-5) is six. Reaching six needs the redistribution
# 7a=7b-11a=12 -> 7a-7b=11a-12, whose middle edge gains a bond while its two
# ends gain hydrogen, so an edge-deletion-only delta cannot express it.
#
# The missing hydrogen used to be absorbed by citing 12H, which is why the
# wrong name looked self-consistent: indole's N-1 carries hydrogen in
# isolation, and carried onto N-12 it names a seven-double-bond parent this
# molecule does not have.
@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_the_hydrogenated_parent_is_named_from_its_own_reference():
    expected = "6,7,7a,12,12a,12b-hexahydroindolo[2,3-a]quinolizine"
    named = name_smiles(INDOLOQUINOLIZINE)

    assert named == expected
    assert verify_with_opsin(named, INDOLOQUINOLIZINE).status == "matched"


# P-14.7.1 and P-14.7.2 keep three hydrogen roles apart, and a name may carry
# all three at once. Ordinary indicated hydrogen selects a realization of the
# unmodified mancude parent; added indicated hydrogen is justified by a named
# operation on that parent and need not preserve its maximum; hydro states the
# remaining hydrogenation. The engine already renders all four of these
# correctly, so they are the guard for any future gate on the ordinary role:
# an added-H site must never be fed to a reference-feasibility test, and an
# ordinary role must survive a suffix replacing the hydrogen it stands for -
# 1H-inden-1-one keeps its 1H although the carbonyl carbon ends with no
# hydrogen at all.
ROLE_CONTROLS = (
    pytest.param("1H-inden-1-one", id="ordinary-only-surviving-a-suffix"),
    pytest.param("quinolin-2(1H)-one", id="added-only-owned-by-the-ketone"),
    pytest.param("pyrazine-2,3-dione", id="paired-groups-need-no-citation"),
    pytest.param("3,3a-dihydro-1H-indene-1,4(2H)-dione", id="all-three-roles-in-one-name"),
)


@pytest.mark.skipif(not opsin_available(), reason="the reference structures come from OPSIN")
@pytest.mark.parametrize("target", ROLE_CONTROLS)
def test_the_three_hydrogen_roles_stay_apart(target):
    from py2opsin import py2opsin

    smiles = py2opsin(target)
    assert smiles, f"OPSIN could not build {target}"
    assert name_smiles(smiles) == target


def test_the_reference_is_still_reached_by_citing_two_hydrogens(monkeypatch):
    """The total is right; the decomposition is not, and that is the next defect.

    P-25.7.1.1 wants the maximum the completed system admits, which here is
    eight with no citation available, so all six hydrogens are hydro. The plan
    instead cites 6H and 12H, which costs the reference its eighth bond, and
    the renderer then folds both citations back into the hydro prefix. Six
    hydrogens are reported either way, so the name above is right - but it is
    right by arriving at the correct total, not by naming the parent the
    molecule has.
    """

    mol = read_rdkit_mol(Chem.MolFromSmiles(INDOLOQUINOLIZINE))
    core = max(find_ring_systems(mol), key=lambda system: len(system.atoms))
    plan = plan_fusion_parent(mol, core.atoms, mode=FusionMode.GENERAL).plan
    unconstrained = parent_bond_model(plan.abstract_parent_graph)

    assert unconstrained.maximum_non_cumulative_double_bonds == 8
    assert plan.bond_model.maximum_non_cumulative_double_bonds == 7, "guard: records the citation's cost"
    assert [str(locant) for locant in plan.indicated_hydrogens] == ["6", "12"]


@pytest.mark.xfail(strict=True, reason="the parent is still reached by citing hydrogen the reference can pair")
def test_the_derivative_ledger_reconstructs_the_target():
    mol = read_rdkit_mol(Chem.MolFromSmiles(INDOLOQUINOLIZINE))
    core = max(find_ring_systems(mol), key=lambda system: len(system.atoms))
    plan = plan_fusion_parent(mol, core.atoms, mode=FusionMode.GENERAL).plan
    locants = {atom: str(locant) for atom, locant in plan.numbering.input_locant_maps[0]}
    hydro = sorted(
        {locants[atom] for operation in plan.derivative_state.hydro_operations for atom in operation.atom_ids}
    )

    assert hydro == ["12", "12a", "12b", "6", "7", "7a"]
