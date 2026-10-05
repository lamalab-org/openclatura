"""Topology-completeness checks for generic chalcogen derivative perception."""

import openclatura as oc
from openclatura.chalcogen_roles import DerivativeKind, FunctionalFamily
from openclatura.graph_io import read_smiles
from openclatura.perception import perceive_groups


def _descriptors(smiles: str):
    return [group.descriptor for group in perceive_groups(read_smiles(smiles)) if group.descriptor is not None]


def test_acyl_ester_does_not_steal_a_colocated_amide_nitrogen():
    descriptors = _descriptors("NC(=O)SC")

    assert any(
        descriptor.family is FunctionalFamily.ACYL and descriptor.derivative is DerivativeKind.AMIDE
        for descriptor in descriptors
    )
    assert not any(descriptor.derivative is DerivativeKind.ESTER for descriptor in descriptors)
    assert oc.name("NC(=O)SC").name == "1-(methylsulfanyl)formamide"


def test_acyl_halide_does_not_steal_a_colocated_amide_nitrogen():
    descriptors = _descriptors("NC(=O)Cl")

    assert any(descriptor.derivative is DerivativeKind.AMIDE for descriptor in descriptors)
    assert not any(descriptor.derivative is DerivativeKind.ACID_HALIDE for descriptor in descriptors)


def test_central_acid_requires_a_neutral_amide_nitrogen():
    descriptors = _descriptors("CS(=O)(=O)[NH3+]")

    assert not any(descriptor.family is FunctionalFamily.CENTRAL_ACID for descriptor in descriptors)


def test_central_acid_rejects_unowned_center_neighbors():
    descriptors = _descriptors("CS(=O)(N)Cl")

    assert not any(descriptor.family is FunctionalFamily.CENTRAL_ACID for descriptor in descriptors)
