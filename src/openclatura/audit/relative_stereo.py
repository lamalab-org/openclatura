"""Exact ring ``cis``/``trans`` determination from tetrahedral parities.

A name can pin a ring's relative configuration with a single word — ``cis-3-
(methylsulfonyl)cyclobutyl`` — instead of per-atom ``R``/``S``.  Confirming such
a name needs an oracle that says, of the *input*, whether two ring substituents
sit on the same face.

The obvious route (embed in 3D, compare against the mean ring plane) is wrong:
a chair cyclohexane puts a cis pair axial/equatorial and straddles that plane,
and the answer varies with the embedding seed.  So this module works purely
combinatorially, from the tetrahedral parities the input SMILES already carries.

For a ring atom ``A`` with ring neighbours ``P`` (predecessor) and ``Q``
(successor) under one fixed traversal direction, plus exocyclic substituent
``S``, the handedness of the ordered triple ``(P, Q, S)`` says which face ``S``
points to.  Because the traversal direction is shared, two ring atoms carry
their substituents on the same face exactly when those handednesses agree — so
``cis`` is parity equality and ``trans`` is parity inequality, whatever the ring
size and however far apart the two atoms are.
"""

from __future__ import annotations

from rdkit import Chem


def ring_face_relation(rdmol: Chem.Mol, atom_a: int, atom_b: int) -> str | None:
    """Return ``"cis"``, ``"trans"``, or ``None`` for the two exocyclic
    substituents borne by ``atom_a`` and ``atom_b``.

    ``None`` means *not determinable* — the atoms share no ring, either lacks an
    assigned tetrahedral parity, or either carries other than exactly one
    exocyclic heavy substituent — and the caller must abstain rather than guess.
    """

    try:
        mol = Chem.AddHs(Chem.Mol(rdmol))
    except Exception:
        return None
    ring = _shared_ring(mol, atom_a, atom_b)
    if ring is None:
        return None

    position = {atom: i for i, atom in enumerate(ring)}
    size = len(ring)
    parities = []
    for atom_idx in (atom_a, atom_b):
        substituent = _sole_exocyclic_substituent(mol, atom_idx, ring)
        if substituent is None:
            return None
        # One fixed traversal direction, shared by both centres.
        predecessor = ring[(position[atom_idx] - 1) % size]
        successor = ring[(position[atom_idx] + 1) % size]
        parity = _face_parity(mol.GetAtomWithIdx(atom_idx), predecessor, successor, substituent)
        if parity is None:
            return None
        parities.append(parity)
    return "cis" if parities[0] == parities[1] else "trans"


def _shared_ring(mol: Chem.Mol, atom_a: int, atom_b: int) -> tuple[int, ...] | None:
    """The smallest ring containing both atoms, or ``None``.

    Smallest so that a fused system resolves to the one ring whose traversal
    actually relates the two centres.
    """

    rings = [r for r in mol.GetRingInfo().AtomRings() if atom_a in r and atom_b in r]
    return min(rings, key=len) if rings else None


def _sole_exocyclic_substituent(mol: Chem.Mol, atom_idx: int, ring: tuple[int, ...]) -> int | None:
    """The atom's single exocyclic heavy neighbour, or ``None`` if it has none or
    several (with several there is no one face to speak of)."""

    outside = [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() not in ring and neighbor.GetAtomicNum() > 1
    ]
    return outside[0] if len(outside) == 1 else None


def _face_parity(atom: Chem.Atom, predecessor: int, successor: int, substituent: int) -> int | None:
    """Handedness (``+1``/``-1``) of ``(predecessor, successor, substituent)`` at a
    tetrahedral centre, or ``None`` when the centre carries no parity.

    RDKit's chiral tag is stated against the atom's own neighbour order, so the
    tag is converted to our chosen order by the sign of the permutation between
    them.  Hydrogens must already be explicit for the neighbour list to be the
    full four.
    """

    tag = atom.GetChiralTag()
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        base = 1
    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        base = -1
    else:
        return None
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    if len(neighbors) != 4:
        return None
    remainder = [n for n in neighbors if n not in (predecessor, successor, substituent)]
    if len(remainder) != 1:
        return None
    return base * _permutation_sign(neighbors, [predecessor, successor, substituent, remainder[0]])


def _permutation_sign(source: list[int], target: list[int]) -> int:
    """Sign of the permutation carrying ``source`` onto ``target``."""

    index = {value: i for i, value in enumerate(source)}
    permutation = [index[value] for value in target]
    sign = 1
    seen = [False] * len(permutation)
    for start in range(len(permutation)):
        if seen[start]:
            continue
        length, cursor = 0, start
        while not seen[cursor]:
            seen[cursor] = True
            cursor = permutation[cursor]
            length += 1
        if length % 2 == 0:  # an even-length cycle is an odd permutation
            sign = -sign
    return sign


__all__ = ["ring_face_relation"]
