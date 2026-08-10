"""Input-order-independent atom ranking.

Naming has to break ties.  Two candidate parent chains can survive every
seniority criterion the Blue Book gives -- the rules run out before the
candidates do -- and something still has to choose one.  Whatever does the
choosing decides the name, so it must be a property of the *structure*: a
tiebreak read off atom indices makes the name depend on where the input SMILES
happened to start, and the same compound then gets two names.

This module computes an equivalence-class rank per atom by iterative refinement
(the classical Morgan relaxation).  Atoms that land in the same class are
indistinguishable by any local graph property, so a tiebreak that treats them as
interchangeable is not making an arbitrary choice -- either one spells the same
name.  Atoms in different classes are ordered by an invariant built from the
element, charge, hydrogen count and bonding, never from the input order.

Refinement equivalence is a slightly coarser relation than graph automorphism:
it can call two atoms equivalent that a full canonicalisation would separate.
That is the safe direction for this use -- it means residual ties are reported
as ties rather than resolved by accident -- and the pathological cases where the
two differ do not arise in molecular graphs of nomenclatural interest.
"""

from .molecule import Molecule


def _initial_invariant(mol: Molecule, idx: int) -> tuple:
    """Local properties of one atom, all independent of how it was indexed."""

    atom = mol.atoms[idx]
    bond_orders = sorted(_order(mol, idx, neighbour) for neighbour in mol.get_neighbors(idx))
    return (
        atom.symbol,
        atom.charge,
        atom.isotope or 0,
        mol.degree(idx),
        atom.total_h_count,
        bool(atom.is_aromatic),
        tuple(bond_orders),
    )


def canonical_ranks(mol: Molecule) -> dict[int, int]:
    """Map each atom index to its refinement-equivalence rank.

    Equal ranks mean the atoms are interchangeable; the ordering between
    different ranks is deterministic and depends only on the structure.
    """

    cached = mol._canonical_rank_cache
    if cached is not None:
        return cached

    invariants = {idx: _initial_invariant(mol, idx) for idx in mol.atoms}
    ranks = _ranks_from_invariants(invariants)

    # Refine until the partition stops getting finer.  Each round replaces an
    # atom's invariant with its own rank plus the multiset of its neighbours',
    # so information spreads one bond further per round; the partition can only
    # get finer, and it is bounded by the atom count.
    previous_classes = len(set(ranks.values()))
    for _ in range(len(mol.atoms)):
        refined = {}
        for idx in mol.atoms:
            neighbour_key = sorted(
                (_order(mol, idx, neighbour), ranks[neighbour]) for neighbour in mol.get_neighbors(idx)
            )
            refined[idx] = (ranks[idx], tuple(neighbour_key))
        ranks = _ranks_from_invariants(refined)
        classes = len(set(ranks.values()))
        if classes == previous_classes:
            break
        previous_classes = classes

    mol._canonical_rank_cache = ranks
    return ranks


AROMATIC_ORDER = 4


def _order(mol: Molecule, u: int, v: int) -> int:
    """Bond order, with every aromatic bond given the same value.

    A Kekule structure assigns alternating single and double bonds around a
    benzene ring, so raw orders would make the ring carbons look inequivalent --
    and *which* ones look single depends on where the input started.  Collapsing
    aromatic bonds to one value keeps the ring's symmetry visible.
    """

    bond = mol.get_bond(u, v)
    if bond is None:
        return 0
    if mol.atoms[u].is_aromatic and mol.atoms[v].is_aromatic:
        return AROMATIC_ORDER
    return bond.order


def _ranks_from_invariants(invariants: dict[int, tuple]) -> dict[int, int]:
    """Number the distinct invariants from 0, ordered by the invariant itself."""

    ordering = {key: rank for rank, key in enumerate(sorted(set(invariants.values())))}
    return {idx: ordering[key] for idx, key in invariants.items()}


def path_rank_key(mol: Molecule | None, path: list[int]) -> tuple[int, ...]:
    """An order-invariant key for a candidate parent skeleton.

    Sorted, so two candidates related by a symmetry of the molecule produce the
    same key and compare equal -- which is correct, since they spell the same
    name.
    """

    if mol is None:
        return tuple(sorted(path))
    ranks = canonical_ranks(mol)
    return tuple(sorted(ranks.get(idx, -1) for idx in path))
