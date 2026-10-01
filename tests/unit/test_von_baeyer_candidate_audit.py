"""A ranked von Baeyer candidate is a preference, not a proof that it rebuilds.

Each molecule here names at the branch point and stopped naming later, because
the one candidate the ranking put first failed its own reconstruction audit and
the parent was abandoned with it. They come from the 100K PubChem corpus, where
ten such skeletons made the benchmark refuse to report a timing at all.
"""

import pytest

from openclatura import chains, name_smiles
from openclatura.graph_io import read_smiles
from openclatura.graph_kernel import biconnected_edge_components
from openclatura.polycycle_topology import build_von_baeyer_numbering

# The ranking used to hand back only its first choice. These skeletons need a
# later one: a differently placed main ring, the mirrored traversal, or the
# atom order rather than the canonical ranks.
NEEDS_A_LATER_CANDIDATE = [
    "Cn1cc2c(n1)CN(C1CC1)C(=O)CCCn1cc(c(C3CC3)n1)C1=CCc3ncc-2nc31",
    "CN1CCN(C)C(=O)c2cc[nH]c2/C=C2\\C(=O)Nc3ccc(nc32)-c2cnn(C)c2C1",
    "CC=C1C=C(C)C2Oc3c(cc(O)cc3C(OC3CCC(O)C(C)O3)C2C)NC(=O)C(C)C1=O",
    "O=C1OC(=O)C23CC12C1CCCCC1C12CC13C(=O)OC2=O",
    "O=C1CCC23CCCCCCC12c1ccccc13",
    "O=C1NCC(C(F)F)Oc2ccc(F)cc2CNc2ccn3ncc1c3n2",
    "CC1C[C@H]2c3cc(F)cnc3CCC3(CC3)NC(=O)c3cnn4ccc(nc34)N2C1",
    "C=C1N[C@@H](CO)COc2ccc(F)cc2[C@@H](C)Nc2ccn3ncc1c3n2",
    "CN1CCCCOc2ccnc3[nH]c(=O)n(c23)-c2ccc3nccc1c3n2",
    "CC(=O)Oc1cc2ccc1Oc1ccc(cc1)CCC(OC(C)=O)CCCC2",
]


@pytest.mark.parametrize("smiles", NEEDS_A_LATER_CANDIDATE)
def test_polycyclic_skeletons_are_named_not_abandoned(smiles):
    assert name_smiles(smiles), "the parent was abandoned when its first candidate failed to reconstruct"


def _largest_ring_block(smiles: str):
    molecule = read_smiles(smiles)
    edges = [(bond.u, bond.v) for bond in molecule.bonds.values()]
    blocks = [block for block in biconnected_edge_components(list(molecule.atoms), edges) if len(block) >= 2]
    blocks.sort(key=len, reverse=True)
    atoms = {atom for edge in blocks[0] for atom in edge}
    return molecule, atoms, {tuple(sorted(edge)) for edge in blocks[0]}


@pytest.mark.parametrize("smiles", NEEDS_A_LATER_CANDIDATE[:4])
def test_some_candidate_reconstructs_even_when_the_first_does_not(smiles):
    """The generator must reach a candidate whose descriptor rebuilds the graph."""

    molecule, atoms, edges = _largest_ring_block(smiles)
    order = sorted(atoms)
    identity = {atom: position for position, atom in enumerate(order)}
    relabelled = {tuple(sorted((identity[u], identity[v]))) for u, v in edges}

    reconstructing = 0
    for index, (descriptor, paths) in enumerate(
        chains.get_von_baeyer_descriptor_candidates(set(identity.values()), relabelled)
    ):
        if index >= 64 or descriptor is None:
            break
        restored = [[order[position] for position in path] for path in paths]
        if any(build_von_baeyer_numbering(descriptor, path, frozenset(edges), molecule).audit_ok for path in restored):
            reconstructing += 1
    assert reconstructing, "no candidate in the ranking rebuilds this skeleton"


def test_both_traversals_of_a_candidate_are_offered():
    """A bridge interior numbered from the wrong end describes a different graph.

    The two traversals run the bridges from opposite ends, and the preferred
    superscript sequence does not always carry the one that reconstructs, so
    both have to reach the audit.
    """

    _, atoms, edges = _largest_ring_block("O=C1CCC23CCCCCCC12c1ccccc13")
    order = sorted(atoms)
    identity = {atom: position for position, atom in enumerate(order)}
    relabelled = {tuple(sorted((identity[u], identity[v]))) for u, v in edges}
    rendered = [
        descriptor
        for index, (descriptor, _) in enumerate(
            chains.get_von_baeyer_descriptor_candidates(set(identity.values()), relabelled)
        )
        if index < 2
    ]
    assert len(rendered) == 2


def test_the_preferred_candidate_is_still_offered_first():
    """Auditing must not reorder the ranking, only skip what cannot rebuild."""

    _, atoms, edges = _largest_ring_block("C1CC2CCC1C2")
    order = sorted(atoms)
    identity = {atom: position for position, atom in enumerate(order)}
    relabelled = {tuple(sorted((identity[u], identity[v]))) for u, v in edges}
    first = next(iter(chains.get_von_baeyer_descriptor_candidates(set(identity.values()), relabelled)))
    assert first[0] == chains.get_von_baeyer_descriptor_and_path(set(identity.values()), relabelled)[0]
    assert name_smiles("C1CC2CCC1C2") == "bicyclo[2.2.1]heptane"
