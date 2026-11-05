"""Large fused faces use topology proofs rather than a ring-size split."""

import pytest

from openclatura.fusion.faces import select_bounded_face_model, typed_face_model
from openclatura.fusion.layout import intrinsic_fused_layouts
from openclatura.molecule import Molecule


def _multiport_system(
    ring_size: int,
    ports: tuple[int, ...],
    *,
    atom_ids: tuple[int, ...] | None = None,
) -> tuple[Molecule, object]:
    cycles = [tuple(range(ring_size))]
    next_atom = ring_size
    for port in ports:
        outside = tuple(range(next_atom, next_atom + 3))
        next_atom += 3
        cycles.append((port, (port + 1) % ring_size, *outside))
    labels = atom_ids or tuple(range(next_atom))
    assert len(labels) == next_atom
    relabelled = [tuple(labels[atom] for atom in cycle) for cycle in cycles]

    mol = Molecule()
    for atom in labels:
        mol.add_atom("C", idx=atom)
    edges = {tuple(sorted((left, right))) for cycle in relabelled for left, right in zip(cycle, cycle[1:] + cycle[:1])}
    for bond_id, (left, right) in enumerate(sorted(edges)):
        mol.add_bond(left, right, idx=bond_id)
    bounded = select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None and bounded.audit.ok
    return mol, typed_face_model(mol, bounded)


def _geometry_signatures(layouts) -> set[tuple]:
    return {
        (
            layout.orientation_score,
            tuple(sorted(shape for _, shape in layout.face_shapes)),
            tuple(sorted((x, y) for _, x, y in layout.atom_positions)),
        )
        for layout in layouts
    }


@pytest.mark.parametrize(
    ("ring_size", "ports"),
    [
        (9, (0, 3, 6)),
        (10, (0, 3, 6)),
        (12, (0, 3, 6, 9)),
        (15, (0, 3, 6, 9, 12)),
    ],
)
def test_large_multiport_faces_receive_complete_audited_geometry(ring_size, ports):
    _mol, model = _multiport_system(ring_size, ports)

    layouts = intrinsic_fused_layouts(model)

    assert layouts
    expected_atoms = {atom for face in model.faces for atom in face.atom_cycle}
    assert all({atom for atom, _, _ in layout.atom_positions} == expected_atoms for layout in layouts)
    assert all({face for face, _ in layout.face_shapes} == {face.id for face in model.faces} for layout in layouts)
    assert all("complete original face geometry audited" in layout.audit_evidence for layout in layouts)


@pytest.mark.parametrize(("ring_size", "ports"), [(9, (0, 3, 6)), (15, (0, 3, 6, 9, 12))])
def test_large_multiport_geometry_is_invariant_to_atom_identifiers(ring_size, ports):
    first_mol, first_model = _multiport_system(ring_size, ports)
    count = len(first_mol.atoms)
    arbitrary_ids = tuple(1000 + 37 * index for index in reversed(range(count)))
    _second_mol, second_model = _multiport_system(ring_size, ports, atom_ids=arbitrary_ids)

    first = intrinsic_fused_layouts(first_model)
    second = intrinsic_fused_layouts(second_model)

    assert first and second
    assert _geometry_signatures(first) == _geometry_signatures(second)
