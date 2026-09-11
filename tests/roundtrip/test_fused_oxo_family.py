"""Sweep a whole structural family for names that do not round-trip.

The engine's hydrogen accounting is proved per molecule, so a change to one of
its guards can be correct on the molecule that motivated it and wrong across
the family the guard governs -- and the rest of the suite will not notice,
because it pins individual names rather than families.

This module enumerates one such family from first principles: ortho-fused
bicycles carrying an oxo on a carbon next to a ring-fusion atom, swept over ring
size, heteroatom placement, and every choice of parent pi bonds. Nothing here is
drawn from a dataset, so it stays an independent check rather than a snapshot of
current behaviour, and OPSIN is the oracle.

``tests/data/fused_oxo_family_known_failures.json`` pins the members that do not
round-trip today. A member outside that list failing is a regression and fails
this test. Members inside it are open defects; when one starts passing, drop it
from the file.
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
# The full family is ~13.5k molecules and takes a few minutes. The default slice
# keeps the ordinary suite quick; set the variable to sweep everything.
DEFAULT_SAMPLE = 2500
KNOWN_FAILURES = Path(__file__).parents[1] / "data" / "fused_oxo_family_known_failures.json"


def _skeleton(size_a: int, size_b: int) -> tuple[int, list[tuple[int, int]]]:
    """Atoms 0 and 1 are the shared fusion bond; the rest close each ring."""

    bonds = [(0, 1)]
    nxt = 2
    for size in (size_a, size_b):
        path = [1] + list(range(nxt, nxt + size - 2)) + [0]
        nxt += size - 2
        bonds += [(path[index], path[index + 1]) for index in range(len(path) - 1)]
    return nxt, bonds


def _neighbours(bonds: list[tuple[int, int]], atom: int) -> set[int]:
    return {site for bond in bonds if atom in bond for site in bond} - {atom}


def _matchings(bonds: list[tuple[int, int]], count: int) -> list[tuple[tuple[int, int], ...]]:
    """Every set of pairwise disjoint ring bonds -- the parent pi-bond choices."""

    found: list[tuple[tuple[int, int], ...]] = [()]
    for size in range(1, count // 2 + 1):
        for combo in combinations(bonds, size):
            used = [atom for bond in combo for atom in bond]
            if len(set(used)) == len(used):
                found.append(combo)
    return found


def _build(count: int, bonds, elements, matching, oxo: int) -> str | None:
    mol = Chem.RWMol()
    for index in range(count):
        mol.AddAtom(Chem.Atom(elements.get(index, "C")))
    doubles = set(matching)
    for bond in bonds:
        mol.AddBond(bond[0], bond[1], Chem.BondType.DOUBLE if bond in doubles else Chem.BondType.SINGLE)
    oxygen = mol.AddAtom(Chem.Atom("O"))
    mol.AddBond(oxo, oxygen, Chem.BondType.DOUBLE)
    try:
        built = mol.GetMol()
        Chem.SanitizeMol(built)
    except Exception:
        return None
    smiles = Chem.MolToSmiles(built)
    # Ordinary-valence neutral species only: a bracket atom in the canonical
    # SMILES means a charge, a radical or a hypervalent centre such as [SH],
    # which are their own naming problem and not this family.
    return None if "[" in smiles else smiles


def enumerate_family() -> list[str]:
    """Every member of the family, deterministically ordered."""

    seen: set[str] = set()
    for size_a, size_b in RING_SIZES:
        count, bonds = _skeleton(size_a, size_b)
        atoms = list(range(count))
        fusion = {0, 1}
        oxo_sites = [a for a in atoms if a not in fusion and _neighbours(bonds, a) & fusion]
        hetero_slots = [a for a in atoms if a not in fusion]
        for hetero_count in range(MAX_HETERO + 1):
            for positions in combinations(hetero_slots, hetero_count):
                for symbols in product(HETERO, repeat=hetero_count):
                    elements = dict(zip(positions, symbols))
                    for oxo in oxo_sites:
                        if oxo in elements:
                            continue
                        for matching in _matchings([b for b in bonds if oxo not in b], count):
                            smiles = _build(count, bonds, elements, matching, oxo)
                            if smiles is not None:
                                seen.add(smiles)
    return sorted(seen)


def _sample(family: list[str]) -> list[str]:
    if os.environ.get("OPENCLATURA_RUN_FULL_FUSED_OXO_SWEEP"):
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
    assert len(family) == 13507, "the generator changed; regenerate the known-failure baseline"
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
