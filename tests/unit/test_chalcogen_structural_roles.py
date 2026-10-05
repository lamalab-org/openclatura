"""Structural tests for composable chalcogen roles."""

import pytest

from openclatura.chalcogen_roles import AcylLeavingGroup, DerivativeKind, FunctionalFamily
from openclatura.graph_io import read_smiles
from openclatura.perception import perceive_groups


def _groups_in_family(smiles: str, family: FunctionalFamily):
    return [
        group
        for group in perceive_groups(read_smiles(smiles))
        if group.descriptor is not None and group.descriptor.family is family
    ]


def test_repeated_chalcogen_linkages_remain_distinct_perceived_groups():
    linkages = _groups_in_family("CC(OSC)(O[Se]C)C", FunctionalFamily.PEROXIDE)

    assert len(linkages) == 2
    assert len({group.descriptor.centers for group in linkages}) == 2


def test_repeated_nitrogen_chalcogenides_remain_distinct_perceived_groups():
    linkages = _groups_in_family(
        "[O-][N+](C)(C)CC[N+](C)(C)[S-]",
        FunctionalFamily.NITROGEN_CHALCOGENIDE,
    )

    assert len(linkages) == 2
    assert len({group.descriptor.centers for group in linkages}) == 2


@pytest.mark.parametrize(
    ("smiles", "kind", "unit_size"),
    [
        ("CC(=O)N=[N+]=[N-]", AcylLeavingGroup.AZIDE, 3),
        ("CC(=O)C#N", AcylLeavingGroup.CYANIDE, 2),
        ("CC(=O)N#C", AcylLeavingGroup.ISOCYANIDE, 2),
        ("CC(=O)N=C=O", AcylLeavingGroup.ISOCYANATE, 3),
        ("CC(=O)N=C=S", AcylLeavingGroup.ISOTHIOCYANATE, 3),
        ("CC(=O)N=C=[Se]", AcylLeavingGroup.ISOSELENOCYANATE, 3),
        ("CC(=O)N=C=[Te]", AcylLeavingGroup.ISOTELLUROCYANATE, 3),
    ],
)
def test_acyl_pseudohalide_owns_complete_leaving_unit(smiles: str, kind: AcylLeavingGroup, unit_size: int):
    matches = [
        group
        for group in perceive_groups(read_smiles(smiles))
        if group.descriptor is not None
        and group.descriptor.derivative is DerivativeKind.ACYL_PSEUDOHALIDE
    ]

    assert len(matches) == 1
    descriptor = matches[0].descriptor
    assert descriptor.leaving_group is not None
    assert descriptor.leaving_group.kind is kind
    assert len(descriptor.leaving_group.atom_ids) == unit_size
    assert len(descriptor.leaving_group.bond_ids) == unit_size
    assert set(descriptor.leaving_group.atom_ids) <= matches[0].atoms_involved
    assert set(descriptor.leaving_group.bond_ids) <= matches[0].bond_ids
