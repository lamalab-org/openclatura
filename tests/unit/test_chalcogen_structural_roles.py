"""Structural tests for composable chalcogen roles."""

from openclatura.chalcogen_roles import FunctionalFamily
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
