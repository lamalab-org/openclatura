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


def _carbon_cycle(atom_count: int) -> Molecule:
    mol = Molecule()
    for atom_id in range(atom_count):
        mol.add_atom("C", idx=atom_id)
    for bond_id, atom_id in enumerate(range(atom_count), start=1):
        mol.add_bond(atom_id, (atom_id + 1) % atom_count, idx=bond_id)
    return mol


def test_large_polycycle_uses_confirmed_fusion_before_descriptor_search(monkeypatch):
    mol = _fused_triangle_chain(9)
    expected_path = list(mol.atoms)

    monkeypatch.setattr(
        "openclatura.chains._confirmed_fusion_numbering_paths",
        lambda _mol, atoms: [sorted(atoms)],
    )

    def fail_descriptor_search(*_args, **_kwargs):
        raise AssertionError("large audited fusion parent reached descriptor search")

    monkeypatch.setattr("openclatura.chains._polyspiro_or_von_baeyer_candidate", fail_descriptor_search)

    systems = find_ring_systems(mol)

    assert len(systems) == 1
    assert systems[0].is_polycycle
    assert systems[0].paths == [expected_path]
    assert systems[0].polycycle_descriptor is None


def test_small_ring_block_uses_graph_decomposition():
    mol = _fused_triangle_chain(2)

    systems = find_ring_systems(mol)

    assert len(systems) == 1
    assert systems[0].is_bicycle
    assert systems[0].ring_parent is not None
    assert systems[0].ring_parent.descriptor == "bicyclo[1.1.0]"


def test_ring_decomposition_has_no_legacy_atom_limit():
    mol = _carbon_cycle(128)

    systems = find_ring_systems(mol)

    assert len(systems) == 1
    assert not systems[0].is_polycycle
    assert len(systems[0].paths[0]) == 128
    assert set(systems[0].paths[0]) == set(range(128))
