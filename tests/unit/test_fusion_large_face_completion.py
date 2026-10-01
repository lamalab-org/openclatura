"""Large carbon faces complete missing topology, not an unbounded cycle tier."""

import pytest

from openclatura.fusion import faces
from openclatura.molecule import Molecule, edges_within_atoms


def _three_faces(size, reverse=False):
    mol = Molecule()
    labels = {i: 100 - i * 3 if reverse else i for i in range(size + 4)}
    cycles = (tuple(range(size)), (0, 1, size), (4, 5, size + 1, size + 2, size + 3))
    for atom, label in labels.items():
        mol.add_atom("O" if atom == size else "N" if atom == size + 2 else "C", idx=label)
    edges = {tuple(sorted((labels[a], labels[b]))) for cycle in cycles for a, b in zip(cycle, cycle[1:] + cycle[:1])}
    for a, b in sorted(edges):
        mol.add_bond(a, b)
    return mol, labels


@pytest.mark.parametrize("size", [9, 12, 20])
@pytest.mark.parametrize("reverse", [False, True])
def test_large_face_with_heterocyclic_neighbors_has_exact_topology(size, reverse):
    mol, labels = _three_faces(size, reverse)
    model = faces.select_bounded_face_model(mol, mol.atoms)
    assert model is not None and model.audit.ok
    assert sorted(len(face.atoms) for face in model.faces) == [3, 5, size]
    assert model.cycle_rank == 3
    assert model.audit.reconstructed_edges == frozenset(edges_within_atoms(mol, set(mol.atoms)))
    assert next(face for face in model.faces if len(face.atoms) == size).edges == frozenset(
        tuple(sorted((labels[i], labels[(i + 1) % size]))) for i in range(size)
    )


def test_completion_does_not_expand_explicit_small_ring_search():
    mol, _ = _three_faces(12)
    assert faces.select_bounded_face_model(mol, mol.atoms, max_ring_size=8) is None


@pytest.mark.parametrize("change", ["hetero", "charge", "oversize"])
def test_unsupported_missing_face_is_not_invented(change):
    mol, labels = _three_faces(21 if change == "oversize" else 12)
    if change == "hetero":
        mol.update_atom(labels[2], symbol="N")
    elif change == "charge":
        mol.update_atom(labels[2], charge=-1)
    assert faces.select_bounded_face_model(mol, mol.atoms) is None


def test_completion_is_edge_seeded_and_budgeted(monkeypatch):
    mol, _ = _three_faces(12)
    small = faces.enumerate_chordless_cycles(mol)
    edges = frozenset(edges_within_atoms(mol, set(mol.atoms)))
    missing = edges - frozenset(edge for cycle in small for edge in cycle.edges)
    original = faces._enumerate_from_start
    seeds = []

    def record(start, *args, **kwargs):
        seeds.append((start, kwargs["seed_neighbor"]))
        return original(start, *args, **kwargs)

    monkeypatch.setattr(faces, "_enumerate_from_start", record)
    completed = faces._complete_large_carbon_faces(mol, frozenset(mol.atoms), edges, small, 10000)
    assert set(seeds) == missing
    assert sorted(len(cycle.atoms) for cycle in completed) == [3, 5, 12]
    with pytest.raises(faces.FaceSearchBudgetExceeded, match="large carbon face completion"):
        faces._complete_large_carbon_faces(mol, frozenset(mol.atoms), edges, small, 1)


def test_complete_small_ring_coverage_never_runs_large_search(monkeypatch):
    mol, _ = _three_faces(6)
    small = faces.enumerate_chordless_cycles(mol)

    def forbidden(*args, **kwargs):
        pytest.fail("covered small rings must not trigger large-cycle enumeration")

    monkeypatch.setattr(faces, "_enumerate_from_start", forbidden)
    assert (
        faces._complete_large_carbon_faces(
            mol, frozenset(mol.atoms), frozenset(edges_within_atoms(mol, set(mol.atoms))), small, 1
        )
        == small
    )
