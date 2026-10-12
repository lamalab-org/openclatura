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


def test_unregistered_heteroatom_acyl_link_does_not_claim_an_ester_route():
    descriptors = _descriptors("NSC(=S)N1CCCCC1")

    assert not any(descriptor.derivative is DerivativeKind.ESTER for descriptor in descriptors)
    assert oc.name("NSC(=S)N1CCCCC1").name == "1-((aminosulfanyl)(thioxo)methyl)piperidine"


def test_chalcogen_chain_does_not_override_a_colocated_carbamate():
    for smiles in ("CSOC(=O)NCCCc1ccccc1", "CCN(C(=O)OS)C"):
        groups = perceive_groups(read_smiles(smiles))
        assert not any(
            group.is_principal_candidate
            and group.descriptor is not None
            and group.descriptor.derivative in {DerivativeKind.ACID, DerivativeKind.ESTER}
            and group.descriptor.linker_paths
            and len(group.descriptor.linker_paths[0]) == 3
            for group in groups
        )

    assert oc.name("CSOC(=O)NCCCc1ccccc1").name == "methylsulfanyl (3-phenylpropyl)carbamate"
    assert oc.name("CCN(C(=O)OS)C").name == "sulfanyl ethyl(methyl)carbamate"


def test_acyl_halide_does_not_steal_a_colocated_amide_nitrogen():
    descriptors = _descriptors("NC(=O)Cl")

    assert any(descriptor.derivative is DerivativeKind.AMIDE for descriptor in descriptors)
    assert not any(descriptor.derivative is DerivativeKind.ACID_HALIDE for descriptor in descriptors)


def test_hydrogen_bearing_acyl_ligand_keeps_acid_priority_over_nitrogen():
    descriptors = _descriptors("NC(=O)O")

    assert any(descriptor.derivative is DerivativeKind.ACID for descriptor in descriptors)
    assert oc.name("NC(=O)O").name == "carbamic acid"


def test_heteroatom_link_is_an_ester_only_when_the_center_is_a_carbamate():
    assert oc.name("C#CCC(=O)ON(C)C(=O)CCC").name == "N-((but-3-ynoyl)oxy)-N-methylbutanamide"
    assert oc.name("CN(C)C(=O)ONC(=O)c1ccccc1").name == "benzamido dimethylcarbamate"


def test_cyclic_heteroatom_link_keeps_the_carbonyl_or_thiocarbonyl_in_the_ring():
    assert oc.name("Cc1ccc(-c2nc(=S)o[nH]2)cc1").name == "3-(4-methylphenyl)-2H-1,2,4-oxadiazole-5-thione"
    assert (
        oc.name("NC(CCn1oc(=O)[nH]c1=O)C(=O)O").name
        == "2-(3-amino-4-hydroxy-4-oxobutyl)-1,2,4-oxadiazolidine-3,5-dione"
    )


def test_central_acid_requires_a_neutral_amide_nitrogen():
    descriptors = _descriptors("CS(=O)(=O)[NH3+]")

    assert not any(descriptor.family is FunctionalFamily.CENTRAL_ACID for descriptor in descriptors)


def test_central_acid_rejects_unowned_center_neighbors():
    descriptors = _descriptors("CS(=O)(N)Cl")

    assert not any(descriptor.family is FunctionalFamily.CENTRAL_ACID for descriptor in descriptors)
