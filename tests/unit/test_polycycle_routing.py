import openclatura.chains as chains
from openclatura.chains import find_ring_systems
from openclatura.molecule import Molecule


def _fused_triangle_chain(ring_count: int) -> Molecule:
    mol = Molecule()
    for atom_id in range(ring_count + 2):
        mol.add_atom("C", idx=atom_id)
    edges = {
        tuple(sorted(edge))
        for ring in range(ring_count)
        for edge in ((ring, ring + 1), (ring + 1, ring + 2), (ring, ring + 2))
    }
    for bond_id, (first, second) in enumerate(sorted(edges), start=1):
        mol.add_bond(first, second, idx=bond_id)
    return mol


def test_large_polycycle_uses_confirmed_fusion_before_legacy_search(monkeypatch):
    mol = _fused_triangle_chain(9)
    expected_path = list(mol.atoms)

    monkeypatch.setattr(
        "openclatura.chains._confirmed_fusion_numbering_paths",
        lambda _mol, atoms: [sorted(atoms)],
    )

    def fail_legacy(*_args, **_kwargs):
        raise AssertionError("large audited fusion parent reached legacy cycle enumeration")

    monkeypatch.setattr("openclatura.chains._polyspiro_or_von_baeyer_candidate", fail_legacy)

    systems = find_ring_systems(mol)

    assert len(systems) == 1
    assert systems[0].is_polycycle
    assert systems[0].paths == [expected_path]
    assert systems[0].polycycle_descriptor is None


def test_small_ring_block_preserves_bounded_legacy_cycle_order(monkeypatch):
    mol = _fused_triangle_chain(2)
    calls = 0
    legacy = chains._legacy_small_cycle_blocks

    def counted_legacy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return legacy(*args, **kwargs)

    monkeypatch.setattr(chains, "_legacy_small_cycle_blocks", counted_legacy)

    systems = find_ring_systems(mol)

    assert calls == 1
    assert len(systems) == 1
    assert systems[0].is_bicycle
    assert systems[0].ring_parent is not None
    assert systems[0].ring_parent.descriptor == "bicyclo[1.1.0]"


def test_high_rank_block_never_enumerates_legacy_cycles(monkeypatch):
    mol = _fused_triangle_chain(9)

    def fail_legacy(*_args, **_kwargs):
        raise AssertionError("high-rank block reached bounded legacy cycle enumeration")

    monkeypatch.setattr(chains, "_legacy_small_cycle_blocks", fail_legacy)
    monkeypatch.setattr(
        chains,
        "_confirmed_fusion_numbering_paths",
        lambda _mol, atoms: [sorted(atoms)],
    )

    systems = find_ring_systems(mol)

    assert len(systems) == 1
    assert systems[0].is_polycycle
