"""Graph-built regressions for junction-relative spiro-side unsaturation."""

from copy import deepcopy

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.assembler import assemble_name_raw
from openclatura.assembly_parent import _parent_unsaturation_locants_are_redundant
from openclatura.assembly_spiro import spiro_assembly_from_parts
from openclatura.graph_io import read_rdkit_mol
from openclatura.locant_elision import apply_redundant_locant_elision
from openclatura.spiro_subgraph import plan_graph_spiro_side


def _graph(*, size=5, double_positions=(1,), core="111", hetero=None, branch=None):
    graph = Chem.RWMol()
    for _ in range(5):
        graph.AddAtom(Chem.Atom("C"))
    if core == "111":
        edges = [(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)]
        junction = 2
    else:
        edges = [(0, 2), (2, 3), (3, 1), (0, 4), (4, 1), (0, 1)]
        junction = 2 if core == "210-long" else 4
    for u, v in edges:
        graph.AddBond(u, v, Chem.BondType.SINGLE)
    if hetero is not None:
        graph.GetAtomWithIdx(3).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(hetero))
    ring = [junction] + [graph.AddAtom(Chem.Atom("C")) for _ in range(size - 1)]
    for position, u in enumerate(ring):
        graph.AddBond(
            u,
            ring[(position + 1) % size],
            Chem.BondType.DOUBLE if position in double_positions else Chem.BondType.SINGLE,
        )
    if branch is not None:
        added = graph.AddAtom(Chem.Atom(branch))
        graph.AddBond(ring[1], added, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol(), ring


@pytest.mark.parametrize("size", [5, 6, 7])
@pytest.mark.parametrize("position", [1, 2])
def test_side_preserves_selected_unsaturation_operations(size, position):
    graph, ring = _graph(size=size, double_positions=(position,))
    mol = read_rdkit_mol(graph)
    side = plan_graph_spiro_side(mol, set(ring), ring[0])
    assert side is not None
    parts = side.side_parts
    assert parts.parent_atom_ids_by_locant[side.side_locant] == ring[0]
    (operation,) = parts.unsaturations
    assert operation.atom_ids == {ring[position], ring[position + 1]}
    assert operation.locants == [str(position + 1)]
    assert f"-{position + 1}-en" in side.side_parent_name
    original = deepcopy(parts)
    projected = spiro_assembly_from_parts(parts, side.side_locant)
    assert projected.side_parent_name == side.side_parent_name
    assert parts == original
    assert projected.side_parts.unsaturations == original.unsaturations


def test_distinct_double_bond_positions_produce_distinct_names():
    results = [name_mol(_graph(double_positions=(position,))[0]) for position in (1, 2)]
    assert all(results), results
    assert results[0].name != results[1].name
    assert "cyclopent-2-ene" in results[0].name
    assert "cyclopent-3-ene" in results[1].name


@pytest.mark.parametrize("omit_locants", [False, True])
@pytest.mark.parametrize("previously_elided", [False, True])
def test_renderer_keeps_typed_metadata_without_retained_parent(omit_locants, previously_elided):
    graph, ring = _graph(double_positions=(2,))
    mol = read_rdkit_mol(graph)
    parts = plan_graph_spiro_side(mol, set(ring), ring[0]).side_parts
    parts.omit_redundant_locants = omit_locants
    if previously_elided:
        parts.elided_unsaturation_locants.add("double")
    original = deepcopy(parts)
    rendered = []

    def render(local):
        assert local.is_spiro_component
        assert not local.omit_redundant_locants
        assert local.retained_name is None
        assert local.parent_hydride == original.parent_hydride
        assert local.unsaturations == original.unsaturations
        assert local.unsaturations is not parts.unsaturations
        assert local.unsaturations[0] is not parts.unsaturations[0]
        assert local.parent_atom_ids_by_locant == original.parent_atom_ids_by_locant
        assert local.parent_bond_ids_by_locants == original.parent_bond_ids_by_locants
        assert not _parent_unsaturation_locants_are_redundant(local)
        apply_redundant_locant_elision(local)
        assert not local.elided_unsaturation_locants
        rendered.append(assemble_name_raw(local))
        assert local.retained_name is None
        assert local.unsaturations == original.unsaturations
        return rendered[-1]

    side = spiro_assembly_from_parts(parts, "1", render_parent=render)
    assert rendered == ["cyclopent-3-ene"]
    assert side.side_parent_name == rendered[0]
    assert side.side_parts == original
    assert parts == original
    assert not parts.is_spiro_component


def test_component_flag_does_not_change_isolated_parent_policy():
    graph, ring = _graph()
    mol = read_rdkit_mol(graph)
    parts = plan_graph_spiro_side(mol, set(ring), ring[0]).side_parts
    assert not parts.is_spiro_component
    assert _parent_unsaturation_locants_are_redundant(parts)
    assert assemble_name_raw(deepcopy(parts)) == "cyclopentene"
    local = deepcopy(parts)
    local.is_spiro_component = True
    assert not _parent_unsaturation_locants_are_redundant(local)
    assert assemble_name_raw(local) == "cyclopent-2-ene"
    assert local.unsaturations == parts.unsaturations
    assert local.retained_name is None


CASES = (
    [
        {"double_positions": (position,), "core": core}
        for position in (1, 2)
        for core in ("111", "210-long", "210-short")
    ]
    + [{"double_positions": (position,), "hetero": hetero} for position in (1, 2) for hetero in ("N", "O")]
    + [
        {"size": 6, "double_positions": (2,)},
        {"size": 7, "double_positions": (2,)},
        {"double_positions": (1, 3)},
        {"double_positions": (2,), "branch": "C"},
        {"double_positions": (2,), "branch": "F"},
        {"double_positions": ()},
    ]
)


@pytest.mark.opsin
@pytest.mark.parametrize("options", CASES)
def test_graph_built_spiro_sides_exact_opsin_and_atom_order(options):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph, _ = _graph(**options)
    original = Chem.MolToSmiles(graph)
    names = []
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order))
        assert result, result
        check = verify_with_opsin(result.name, original, standardize_smiles=False)
        assert check.ok, check.to_dict()
        assert check.canonical_roundtrip == original, check.to_dict()
        names.append(result.name)
    assert names[0] == names[1]
