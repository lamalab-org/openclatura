"""Multiplied HW citations must multiply components, not heteroatom prefixes."""

from dataclasses import replace

import pytest
from rdkit import Chem
from unit.test_fusion_descriptor import _build

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.descriptor import render_fusion_name, render_fusion_name_parts
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol


@pytest.fixture(
    params=(
        ("generated-hw:O.C.C", "bisoxireno"),
        ("generated-hw:S.C.C", "bisthiireno"),
        ("generated-hw:O.C.C.C", "bisoxeto"),
        ("furan", "difuro"),
    )
)
def multiplied_case(request):
    component, prefix = request.param
    components = (component, component, "benzene", "pyridine")
    interfaces = (
        ((0, "2"), (2, "1")),
        ((0, "3"), (2, "2")),
        ((1, "2"), (2, "3")),
        ((1, "3"), (2, "4")),
        ((2, "5"), (3, "2")),
        ((2, "6"), (3, "3")),
    )
    return components, interfaces, prefix


def test_multiplied_hw_rendering_preserves_graph_scopes_and_primes(multiplied_case):
    components, interfaces, prefix = multiplied_case
    mol, ast, name = _build(components, interfaces, atomic_components_only=True)
    expected = prefix + "[2',3':1,2;2'',3'':3,4]benzo[5,6-b]pyridine"
    assert name == expected
    registry = fusion_component_registry()
    parts = render_fusion_name_parts(ast, registry, mol=mol)
    assert "".join(part.text for part in parts) == expected
    group = ast.multiplicative_groups[0]
    joins = tuple(join for join in ast.joins if join.attached_occurrence in group.occurrence_ids)
    binding = next(part for part in parts if part.text.startswith("[2',3':"))
    assert binding.atom_ids == frozenset().union(*(join.shared_input_atoms for join in joins))
    assert binding.bond_ids == frozenset().union(*(join.shared_input_bonds for join in joins))
    assert [{locant.prime_depth for locant in join.attached_locants} for join in joins] == [{1}, {2}]

    _, _, renumbered = _build(
        components,
        interfaces,
        atomic_components_only=True,
        atom_id_order=tuple(100 + 7 * i for i in reversed(range(len(mol.atoms)))),
    )
    assert renumbered == expected
    # Registry identifiers are implementation details, not grammar decisions.
    specs = {
        f"component-{i}": replace(registry.spec_for_match(match), key=f"component-{i}")
        for i, match in enumerate(ast.component_occurrences)
    }
    renamed = replace(
        ast,
        component_occurrences=tuple(
            replace(match, spec_key=f"component-{i}") for i, match in enumerate(ast.component_occurrences)
        ),
    )
    assert render_fusion_name(renamed, specs) == expected


def _saturated_scope_smiles(mol, atom_ids):
    """Isolate citation connectivity from bond placement and final numbering."""
    graph = Chem.RWMol()
    positions = {i: graph.AddAtom(Chem.Atom(mol.atoms[i].symbol)) for i in sorted(atom_ids)}
    for bond in mol.bonds.values():
        if bond.u in positions and bond.v in positions:
            graph.AddBond(positions[bond.u], positions[bond.v], Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return Chem.MolToSmiles(graph)


@pytest.mark.opsin
def test_first_order_multiplied_hw_citation_exact_graph_roundtrip(multiplied_case):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    components, _, prefix = multiplied_case
    mol, _, name = _build(
        (*components[:2], "pyridine"),
        (
            ((0, "2"), (2, "2")),
            ((0, "3"), (2, "3")),
            ((1, "2"), (2, "5")),
            ((1, "3"), (2, "6")),
        ),
        atomic_components_only=True,
    )
    assert name.startswith(prefix + "[")
    assert ";" not in name
    check = verify_with_opsin("perhydro" + name, _saturated_scope_smiles(mol, mol.atoms), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip


@pytest.mark.opsin
def test_multiplied_hw_citation_exact_graph_roundtrip(multiplied_case):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    components, interfaces, _ = multiplied_case
    mol, _, name = _build(components, interfaces, atomic_components_only=True)
    check = verify_with_opsin("perhydro" + name, _saturated_scope_smiles(mol, mol.atoms), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip


@pytest.fixture(
    params=(
        pytest.param(
            "CCCCCC(=O)O[C@@H]1[C@@]2(C(C)C)O[C@H]2[C@@H]2O[C@]23[C@]12O[C@H]2C[C@H]1C2=C(CC[C@@]13C)C(=O)OC2",
            id="4582",
        ),
        pytest.param(
            "CC(C)[C@]12O[C@H]1[C@@H]1O[C@]13[C@]1(O[C@H]1C[C@H]1C4=C(C(=O)c5ccccc5)OC(=O)C4CC[C@@]13C)[C@@H]2O",
            id="58492",
        ),
    )
)
def multiplied_pubchem_parent(request):
    graph = Chem.MolFromSmiles(request.param)
    scopes = []
    for ring in graph.GetRingInfo().AtomRings():
        scope = set(ring)
        for previous in scopes[:]:
            if scope & previous:
                scope.update(previous)
                scopes.remove(previous)
        scopes.append(scope)
    atom_ids = max(scopes, key=len)
    mol = read_rdkit_mol(graph)
    result = plan_fusion_parent(mol, atom_ids, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    return graph, mol, atom_ids, result.plan


def test_pubchem_multiplied_parent_uses_unambiguous_component_multiplier(multiplied_pubchem_parent):
    graph, _, _, plan = multiplied_pubchem_parent
    expected = "trisoxireno[2',3':2,3;2'',3'':4,4a;2''',3''':10,10a]phenanthro[7,8-c]furan"
    assert plan.rendered_base_name == expected
    names = [
        name_mol(mol).name
        for mol in (
            graph,
            Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms())))),
        )
    ]
    assert names[0] == names[1]
    assert expected in names[0]


@pytest.mark.opsin
def test_pubchem_multiplied_parent_exact_graph_roundtrip(multiplied_pubchem_parent):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    _, mol, atom_ids, plan = multiplied_pubchem_parent
    check = verify_with_opsin(
        "perhydro" + plan.rendered_base_name,
        _saturated_scope_smiles(mol, atom_ids),
        standardize_smiles=False,
    )
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip
