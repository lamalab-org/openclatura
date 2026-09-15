"""The cycle-space main ring must agree with the walk it stands in for."""

import math
import time

import pytest

from openclatura import chains
from openclatura.graph_io import read_smiles
from openclatura.graph_kernel import biconnected_edge_components


def _largest_ring_block(smiles: str):
    mol = read_smiles(smiles)
    edges = [(bond.u, bond.v) for bond in mol.bonds.values()]
    blocks = [block for block in biconnected_edge_components(list(mol.atoms), edges) if len(block) >= 2]
    blocks.sort(key=len, reverse=True)
    nodes = {atom for edge in blocks[0] for atom in edge}
    return nodes, {tuple(sorted(edge)) for edge in blocks[0]}


def _benzenoid(hexagons: int):
    """A compact cluster of fused hexagons, as a plain graph."""

    centres = [(0, 0)]
    q = r = 0
    ring = 1
    steps = [(1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0)]
    while len(centres) < hexagons:
        q, r = q + 1, r - 1
        for step in steps:
            for _ in range(ring):
                if len(centres) >= hexagons:
                    break
                centres.append((q, r))
                q, r = q + step[0], r + step[1]
            if len(centres) >= hexagons:
                break
        ring += 1
    vertices: dict[tuple[float, float], int] = {}
    edges = set()
    for centre_q, centre_r in centres[:hexagons]:
        x = 1.5 * centre_q
        y = math.sqrt(3) * (centre_r + centre_q / 2)
        corners = []
        for corner in range(6):
            angle = math.radians(60 * corner)
            point = (round(x + math.cos(angle), 5), round(y + math.sin(angle), 5))
            vertices.setdefault(point, len(vertices))
            corners.append(vertices[point])
        for index in range(6):
            edges.add(tuple(sorted((corners[index], corners[(index + 1) % 6]))))
    return set(vertices.values()), edges


def _walk_longest_cycle(nodes, edges) -> int:
    """The length of the longest cycle, by the exhaustive walk."""

    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    longest = 0

    def visit(current, start, path, seen):
        nonlocal longest
        for neighbor in adjacency[current]:
            if neighbor == start and len(path) >= 3:
                longest = max(longest, len(path))
            elif neighbor not in seen:
                visit(neighbor, start, path + [neighbor], seen | {neighbor})

    for node in nodes:
        visit(node, node, [node], {node})
    return longest


def _cycle_space_longest_cycle(nodes, edges) -> int:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    found = chains._largest_cycles_from_cycle_space(set(nodes), set(edges), adjacency)
    return len(found[0]) if found else 0


SYSTEMS = {
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "fluoranthene": "c1ccc2c(c1)-c1cccc3cccc2c13",
    "acenaphthylene": "C1=Cc2cccc3cccc1c23",
    # a bowl: the pentagon is enclosed by hexagons
    "corannulene": "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51",
    "coronene": "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61",
    # cages, where no planar face structure exists at all
    "adamantane": "C1C2CC3CC1CC(C2)C3",
    "cubane": "C1C2C3C1C1C2C3C1",
    "norbornane": "C1CC2CCC1C2",
}


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_cycle_space_main_ring_matches_the_walk(name):
    nodes, edges = _largest_ring_block(SYSTEMS[name])
    assert _cycle_space_longest_cycle(nodes, edges) == _walk_longest_cycle(nodes, edges)


@pytest.mark.parametrize("hexagons", [2, 4, 6, 8, 10, 12])
def test_cycle_space_main_ring_matches_the_walk_on_fused_hexagons(hexagons):
    nodes, edges = _benzenoid(hexagons)
    assert _cycle_space_longest_cycle(nodes, edges) == _walk_longest_cycle(nodes, edges)


def test_cycle_space_reaches_a_block_the_walk_cannot():
    """A block past the walk's budget still yields a main ring, and quickly.

    The walk costs about twice as much per added ring; this one needs order
    1e9 states and ran for 658 s before this path existed. The cycle space has
    one dimension per ring, so the same answer is 2**17 subsets away.
    """

    nodes, edges = _benzenoid(17)
    rank = len(edges) - len(nodes) + 1
    assert rank > 13, "this block must be past the walk's reach to be worth testing"
    started = time.perf_counter()
    descriptor, paths = chains.get_von_baeyer_descriptor_and_path(set(nodes), set(edges))
    assert time.perf_counter() - started < 60
    assert descriptor is not None
    assert descriptor.endswith("]")
    assert paths and len(paths[0]) == len(nodes)
