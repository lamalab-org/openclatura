"""A fused parent cited as a substituent misallocates its hydrogen.

Parent indicated hydrogen, suffix-generated added hydrogen and genuine
hydrogenation are three separate citation mechanisms (P-58.2.3.1.3, .1.4) and
none stands in for another. A compound substituent takes its free valence as
the governing suffix (P-59.1.9), so the parent's indicated hydrogen is spent on
that valence first (P-58.2.3.1.1) and the rest of the saturation - the two oxo
positions and the residual ring carbons - is detachable hydro.

The free valence now claims that hydrogen and the name reads, but the carbons
it vacated are still cited as added hydrogen rather than as the hydro prefix a
compound substituent calls for, so the preferred spelling is not reached yet.

The controls below reach the same shape and are named correctly, because the
assembler supplies a hydro prefix for them; they are here so a change aimed at
the gap cannot quietly take them with it. Their derivative states are
indistinguishable from the broken one at plan level - same empty
hydrogenated_edges, hydro, intrinsic and added-H operations - so a guard that
reads the plan alone cannot separate them.
"""

import pytest
from rdkit import Chem

from openclatura import name_mol, name_smiles, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles

# Needs four saturated ring carbons: 4a and 8 as indicated hydrogen, 4b and 7a
# as added hydrogen consumed by the 5,7-dioxo.
UNAUDITED = "CC(=O)Oc1ccc(N2C(=O)[C@H]3[C@@H](C2=O)C2C=CC=NN2[C@H]3C(=O)c2ccccc2)cc1"

# Same shape, named correctly today: the assembler spells the saturation as a
# hydro prefix rather than needing the combined citation.
CONTROLS = (
    ("C1Cc2ncccc2C1=O", "6,7-dihydro-5H-cyclopenta[b]pyridin-5-one"),
    ("CC1Cc2ncccc2C1=O", "6-methyl-6,7-dihydro-5H-cyclopenta[b]pyridin-5-one"),
    ("C1Cc2ncc(Cl)cc2C1=O", "3-chloro-6,7-dihydro-5H-cyclopenta[b]pyridin-5-one"),
)


def _largest_ring_system(smiles):
    mol = read_smiles(smiles)
    return mol, max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms


@pytest.mark.parametrize(("smiles", "expected"), CONTROLS)
def test_a_hydro_prefix_still_spells_the_consumed_saturation(smiles, expected):
    """These must keep their citation form through any work on the tier."""

    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(("smiles", "expected"), CONTROLS)
@pytest.mark.skipif(not opsin_available(), reason="OPSIN round-trip needs java and py2opsin")
def test_the_controls_round_trip(smiles, expected):
    assert verify_with_opsin(name_smiles(smiles), smiles).status == "matched"


@pytest.mark.parametrize(("smiles", "expected"), CONTROLS)
def test_the_controls_are_order_invariant(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    assert name_mol(mol).name == name_mol(reversed_mol).name


def test_the_unaudited_parent_still_names_and_is_order_invariant():
    """Whatever it is named, the engine must answer and answer consistently."""

    mol = Chem.MolFromSmiles(UNAUDITED)
    reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    named = name_mol(mol)
    assert named.error is None
    assert named.name
    assert named.name == name_mol(reversed_mol).name


def test_the_fusion_plan_accounts_for_every_saturated_carbon():
    """No ring position may carry hydrogen that no operation accounts for."""

    mol, atoms = _largest_ring_system(UNAUDITED)
    # The ring attaches through this nitrogen, which is what a substituent
    # citation tells the planner; without it the parent spends its hydrogen on
    # ring carbons and two positions go unsaid.
    free_valence = next(
        atom
        for atom in atoms
        if mol.atoms[atom].symbol == "N" and any(other not in atoms for other in mol.get_neighbors(atom))
    )
    result = plan_fusion_parent(mol, atoms, mode="audited_pin", free_valence_atom=free_valence)
    plan = getattr(result, "plan", None)
    if plan is None:  # a later tier may decline the parent outright instead
        pytest.skip("the fused parent is no longer confirmed for this system")
    state = plan.derivative_state
    locants = {atom: str(locant) for atom, locant in plan.numbering.input_locant_maps[0]}
    stated = {atom for atom, locant in locants.items() if locant in set(map(str, plan.indicated_hydrogens))}
    assert str(locants[free_valence]) in set(map(str, plan.indicated_hydrogens))
    for operation in tuple(state.hydro_operations) + tuple(state.intrinsic_hydro_operations):
        stated.update(operation.atom_ids)
    stated.update(state.bond_delta.hydrogenated_atom_ids or ())
    stated.update(atom for operation in state.bond_delta.added_hydrogen_operations for atom in operation.atom_ids)
    external = {operation.parent_atom_id for operation in state.external_pi_operations}
    saturated = {
        atom
        for atom in atoms
        if mol.atoms[atom].total_h_count > 0
        and all(mol.get_bond(atom, other).order == 1 for other in mol.get_neighbors(atom) if other in atoms)
    }
    assert sorted(locants[atom] for atom in saturated - stated - external) == []


# The rule-derived preferred prefix: -6-yl is the governing suffix, the
# carbonyls are demoted to 5,7-dioxo, the free valence takes the parent's
# indicated hydrogen as 6H, and the remaining saturation is detachable hydro.
PREFERRED_PREFIX = "5,7-dioxo-4a,4b,5,7,7a,8-hexahydro-6H-pyrrolo[3',4':3,4]pyrrolo[1,2-b]pyridazin-6-yl"


@pytest.mark.skipif(not opsin_available(), reason="OPSIN round-trip needs java and py2opsin")
def test_the_substituted_parent_round_trips():
    """The free valence claims the parent's indicated hydrogen, so it names."""

    assert verify_with_opsin(name_smiles(UNAUDITED), UNAUDITED).status == "matched"


@pytest.mark.xfail(strict=True, reason="the vacated carbons are cited as added hydrogen, not as a hydro prefix")
def test_the_substituted_parent_uses_the_preferred_prefix():
    """Round-tripping is not enough: the citation mechanisms must be right.

    An all-indicated-hydrogen spelling denotes the same structure and OPSIN
    reads it, so a round-trip check alone cannot tell the two apart.
    """

    assert PREFERRED_PREFIX in name_smiles(UNAUDITED)
