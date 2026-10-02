"""How a fused parent cited as a substituent allocates its hydrogen.

Parent indicated hydrogen, suffix-generated added hydrogen and genuine
hydrogenation are three separate citation mechanisms (P-58.2.3.1.3, .1.4) and
none stands in for another. A compound substituent takes its free valence as
the governing suffix (P-59.1.9), so the parent's indicated hydrogen is spent on
that valence first (P-58.2.3.1.1) and the rest of the saturation - the two oxo
positions and the residual ring carbons - is detachable hydro.

Here the indicated hydrogen is spent on ring carbons instead, which leaves the
free valence without one and understates the hydrogenation by four positions,
in a name OPSIN cannot read.

Spending it on the free valence instead was tried in e857651 and reverted: it
reached the preferred prefix here but regressed 52 PubChem molecules of the
same family, because suppressing the carbon citation is what frees the
hydrogen and nothing downstream puts it back - 3,4,5,6-tetrahydroazepino-
[4,5,6-cd]indol-5-yl came out as 3,4-dihydro-5H-azepino[4,5,6-cd]indol-5-yl,
four stated hydrogens becoming three. The xfails below mark what is still
owed.

The controls below reach the same shape and were already named correctly,
because the assembler supplies a hydro prefix for them; they are here so a change aimed at
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

# Needs four saturated ring carbons: 4a and 8 clear of the carbonyls, 4b and 7a
# next to the 5,7-dioxo that took their pi bonds.
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


def test_the_fusion_plan_leaves_two_ring_carbons_unsaid():
    """Pin the defect: 4b and 7a carry hydrogen no operation accounts for."""

    mol, atoms = _largest_ring_system(UNAUDITED)
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    plan = getattr(result, "plan", None)
    if plan is None:  # a later tier may decline the parent outright instead
        pytest.skip("the fused parent is no longer confirmed for this system")
    state = plan.derivative_state
    locants = {atom: str(locant) for atom, locant in plan.numbering.input_locant_maps[0]}
    stated = {atom for atom, locant in locants.items() if locant in set(map(str, plan.indicated_hydrogens))}
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
    assert sorted(locants[atom] for atom in saturated - stated - external) == ["4b", "7a"]


# The rule-derived preferred prefix: -6-yl is the governing suffix, the
# carbonyls are demoted to 5,7-dioxo, the free valence takes the parent's
# indicated hydrogen as 6H, and the remaining saturation is detachable hydro.
PREFERRED_PREFIX = "5,7-dioxo-4a,4b,5,7,7a,8-hexahydro-6H-pyrrolo[3',4':3,4]pyrrolo[1,2-b]pyridazin-6-yl"


@pytest.mark.xfail(strict=True, reason="the free valence does not claim the parent's indicated hydrogen")
@pytest.mark.skipif(not opsin_available(), reason="OPSIN round-trip needs java and py2opsin")
def test_the_substituted_parent_round_trips():
    """Turns green when the allocation is fixed; update this then."""

    assert verify_with_opsin(name_smiles(UNAUDITED), UNAUDITED).status == "matched"


@pytest.mark.xfail(strict=True, reason="the free valence does not claim the parent's indicated hydrogen")
def test_the_substituted_parent_uses_the_preferred_prefix():
    """Round-tripping is not enough: the citation mechanisms must be right.

    An all-indicated-hydrogen spelling denotes the same structure and OPSIN
    reads it, so a round-trip check alone cannot tell the two apart.
    """

    assert PREFERRED_PREFIX in name_smiles(UNAUDITED)
