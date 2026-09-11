"""Sweep fused bicycles at every hydrogenation level for names that round-trip.

The companion sweep in ``test_fused_oxo_family`` always hangs an oxo beside a
ring-fusion atom, so it exercises suffix-driven hydrogen accounting. This one
drops the oxo and varies only the ring: element placement and which parent pi
bonds are present, from the fully mancude system down to the fully saturated
one. That makes it the direct gate on how a fused *parent hydride* is chosen and
how its indicated hydrogen and hydro prefixes are cited.

It is the family that ``component_carbon_h_relocation_scope`` governs. A
component's declared indicated hydrogen (``pyran`` is ``2H-pyran``) must not be
carried into the fused system -- P-25.7.1.3 assigns indicated hydrogen to the
completed ring system -- and when that carbon lands on a ring fusion, holding it
saturated strands every bond incident on it and silently drops a hydrogen from
the name.

Nothing here comes from a dataset, so it stays independent of any evaluation
set, and OPSIN is the oracle. ``tests/data/fused_ring_family_known_failures.json``
pins the members that do not round-trip today; a member outside that list
failing is a regression. When one starts passing, drop it from the file.
"""

import json
import os
from itertools import combinations, product
from pathlib import Path

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available

pytestmark = pytest.mark.opsin

HETERO = ("N", "O", "S")
RING_SIZES = ((5, 5), (5, 6), (6, 6))
MAX_HETERO = 2
# The full family is ~9.4k molecules and takes a couple of minutes. The default
# slice keeps the ordinary suite quick; set the variable to sweep everything.
DEFAULT_SAMPLE = 2000
KNOWN_FAILURES = Path(__file__).parents[1] / "data" / "fused_ring_family_known_failures.json"


def _skeleton(size_a: int, size_b: int) -> tuple[int, list[tuple[int, int]]]:
    """Atoms 0 and 1 are the shared fusion bond; the rest close each ring."""

    bonds = [(0, 1)]
    nxt = 2
    for size in (size_a, size_b):
        path = [1] + list(range(nxt, nxt + size - 2)) + [0]
        nxt += size - 2
        bonds += [(path[index], path[index + 1]) for index in range(len(path) - 1)]
    return nxt, bonds


def _matchings(bonds: list[tuple[int, int]], count: int) -> list[tuple[tuple[int, int], ...]]:
    """Every set of pairwise disjoint ring bonds -- the parent pi-bond choices."""

    found: list[tuple[tuple[int, int], ...]] = [()]
    for size in range(1, count // 2 + 1):
        for combo in combinations(bonds, size):
            used = [atom for bond in combo for atom in bond]
            if len(set(used)) == len(used):
                found.append(combo)
    return found


def _build(count: int, bonds, elements, matching) -> str | None:
    mol = Chem.RWMol()
    for index in range(count):
        mol.AddAtom(Chem.Atom(elements.get(index, "C")))
    doubles = set(matching)
    for bond in bonds:
        mol.AddBond(bond[0], bond[1], Chem.BondType.DOUBLE if bond in doubles else Chem.BondType.SINGLE)
    try:
        built = mol.GetMol()
        Chem.SanitizeMol(built)
    except Exception:
        return None
    smiles = Chem.MolToSmiles(built)
    # Ordinary-valence neutral species only: a bracket atom in the canonical
    # SMILES means a charge, a radical or a hypervalent centre, which are their
    # own naming problem and not this family.
    return None if "[" in smiles else smiles


def enumerate_family() -> list[str]:
    """Every member of the family, deterministically ordered."""

    seen: set[str] = set()
    for size_a, size_b in RING_SIZES:
        count, bonds = _skeleton(size_a, size_b)
        hetero_slots = [atom for atom in range(count) if atom not in {0, 1}]
        for hetero_count in range(MAX_HETERO + 1):
            for positions in combinations(hetero_slots, hetero_count):
                for symbols in product(HETERO, repeat=hetero_count):
                    elements = dict(zip(positions, symbols))
                    for matching in _matchings(bonds, count):
                        smiles = _build(count, bonds, elements, matching)
                        if smiles is not None:
                            seen.add(smiles)
    return sorted(seen)


def _sample(family: list[str]) -> list[str]:
    if os.environ.get("OPENCLATURA_RUN_FULL_FUSED_RING_SWEEP"):
        return family
    step = max(1, len(family) // DEFAULT_SAMPLE)
    return family[::step][:DEFAULT_SAMPLE]


def _canonical(smiles: str) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def test_family_is_large_and_stable():
    family = enumerate_family()
    assert len(family) == 9426, "the generator changed; regenerate the known-failure baseline"
    assert enumerate_family() == family


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
def test_no_member_outside_the_known_failures_loses_its_round_trip():
    from py2opsin import py2opsin

    known = set(json.loads(KNOWN_FAILURES.read_text())["known_failures"])
    members = _sample(enumerate_family())

    names = []
    for smiles in members:
        result = name_mol(Chem.MolFromSmiles(smiles))
        names.append(result.name or "")
    roundtrips = py2opsin([name or "!" for name in names])
    if isinstance(roundtrips, str):
        roundtrips = roundtrips.splitlines()

    regressions = []
    for smiles, name, roundtrip in zip(members, names, roundtrips):
        if _canonical(roundtrip) == _canonical(smiles):
            continue
        if smiles in known:
            continue
        regressions.append((smiles, name or "<no name>"))

    assert not regressions, "names stopped round-tripping:\n" + "\n".join(
        f"  {smiles}\n    {name}" for smiles, name in sorted(regressions)[:25]
    )
