"""Fused bonds come from the biconnected blocks, not from a cycle enumeration."""

import subprocess
import sys

import pytest

from openclatura.graph_io import read_smiles
from openclatura.molecule import edges_within_atoms
from openclatura.polycycle_topology import fused_bonds

# PubChem CID 10876314, a dihydro[60]fullerene.
FULLERENE = (
    "C12C3C4=C5C6=C7C8=C9C(=C61)C1=C2C2=C6C%10=C1C1=C%11C%10=C%10C%12=C%13C%14=C%15C%16=C%17C(=C%18C5=C7C5=C7C%18="
    "C%17C%17=C%15C%15=C%13C%10=C%10C%11=C%11C(=C91)C8=C5C1=C%11C%10=C%15C%17=C71)C4=C1C3=C2C(=C6%12)C%14=C1%16"
)
# PubChem CID 20776421, a saturated polycyclic cage.
CAGE = (
    "C1C2C3C4C5CC6C57C48C39C22C1C1C22C93C84C75C6C6C55C44C33C22C1C1C22C33C44C55C6C6C55C44C33C22C1C1C22C33C44C55C6C6C55"
    "C44C33C22C1CC2C3C4C5C6"
)


def _cycle_count_fused_bonds(mol, atoms):
    """The bonds that lie on two or more simple cycles, by enumerating every cycle."""

    edges = edges_within_atoms(mol, set(atoms))
    adjacency = {atom: set() for atom in atoms}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    cycles = set()

    def visit(current, start, path):
        for neighbor in adjacency[current]:
            if neighbor == start and len(path) >= 3:
                cycles.add(frozenset(tuple(sorted(pair)) for pair in zip(path, path[1:] + path[:1])))
            elif neighbor not in path and neighbor > start:
                visit(neighbor, start, path + [neighbor])

    for atom in atoms:
        visit(atom, atom, [atom])
    counts: dict[tuple[int, int], int] = {}
    for cycle in cycles:
        for edge in cycle:
            counts[edge] = counts.get(edge, 0) + 1
    return tuple(sorted(edge for edge, count in counts.items() if count > 1))


SYSTEMS = {
    "cyclohexane": "C1CCCCC1",
    "naphthalene": "c1ccc2ccccc2c1",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "coronene": "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61",
    "norbornane": "C1CC2CCC1C2",
    "adamantane": "C1C2CC3CC1CC(C2)C3",
    "cubane": "C1C2C3C1C1C2C3C1",
    # blocks that are single rings share no bond with another cycle
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
    "spiro[4.5]decane": "C1CCC2(CC1)CCCC2",
    "dispiro": "C1CC2(C1)CCC1(CC2)CCC1",
    # one fused block next to a plain ring, with a chain between them
    "naphthylbenzyl": "c1ccc(Cc2ccc3ccccc3c2)cc1",
}


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_fused_bonds_match_the_cycle_count(name):
    mol = read_smiles(SYSTEMS[name])
    atoms = frozenset(mol.atoms)
    assert fused_bonds(mol, atoms) == _cycle_count_fused_bonds(mol, atoms)


def test_fused_bonds_of_an_atom_subset():
    # Naphthalene's second ring alone is a single ring: nothing is fused.
    mol = read_smiles("c1ccc2ccccc2c1")
    ring = frozenset({3, 4, 5, 6, 7, 8})
    assert fused_bonds(mol, ring) == ()


@pytest.mark.parametrize("smiles", [FULLERENE, CAGE], ids=["fullerene", "cage"])
def test_fused_bonds_return_where_cycles_cannot_be_enumerated(smiles):
    """These systems have far too many simple cycles to list; finding their
    fused bonds that way never returned (lamalab-org/openclatura#108)."""

    program = """
import sys

from openclatura.graph_io import read_smiles
from openclatura.molecule import edges_within_atoms
from openclatura.polycycle_topology import fused_bonds

mol = read_smiles(sys.argv[1])
atoms = frozenset(mol.atoms)
print(len(fused_bonds(mol, atoms)), len(edges_within_atoms(mol, set(atoms))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, smiles],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    fused, bonds = completed.stdout.split()
    # Every bond of a single polycyclic block is fused.
    assert fused == bonds
