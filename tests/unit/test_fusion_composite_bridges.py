from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.fusion.composite_bridges import composite_bridge_constructions_from_data
from openclatura.fusion.model import FusionMode
from openclatura.fusion.wrappers import (
    NondetachableBridgeKind,
    _audit_bridge_plan,
    _bridge_class,
    _bridge_operations,
    _retained_wrapper_parent,
    plan_bridged_fusion_wrapper,
)
from openclatura.graph_io import read_rdkit_mol
from openclatura.naming_data import load_json_table
from openclatura.opsin_verify import verify_with_opsin

CASES = (
    ("O", (), "epoxymethano", ()),
    ("O", (2, 1), "epoxyprop[1]eno", ("1",)),
    ("O", (1, 2), "epoxyprop[2]eno", ("2",)),
)


def _bridged_quinoline(heteroatom, carbon_orders, endpoints=("5", "2"), *, oxo=False):
    source = Chem.MolFromSmiles("c1ccc2ncccc2c1")
    mol = read_rdkit_mol(source)
    parent = _retained_wrapper_parent(mol, frozenset(mol.atoms))
    atom_by_locant = {locant: atom for atom, locant in parent.locant_maps[0]}
    editable = Chem.RWMol(source)
    path = [editable.AddAtom(Chem.Atom(heteroatom))]
    path.extend(editable.AddAtom(Chem.Atom("C")) for _ in range(len(carbon_orders) + 1))
    for left, right, order in zip(path, path[1:], (1, *carbon_orders)):
        editable.AddBond(left, right, Chem.BondType.values[order])
    editable.AddBond(atom_by_locant[endpoints[0]], path[0], Chem.BondType.SINGLE)
    editable.AddBond(path[-1], atom_by_locant[endpoints[1]], Chem.BondType.SINGLE)
    if oxo:
        oxygen = editable.AddAtom(Chem.Atom("O"))
        editable.AddBond(path[-1], oxygen, Chem.BondType.DOUBLE)
    Chem.SanitizeMol(editable)
    return editable.GetMol(), tuple(path), atom_by_locant


@pytest.mark.parametrize("heteroatom,carbon_orders,prefix,unsaturation", CASES)
@pytest.mark.parametrize("endpoints", [("5", "2"), ("2", "5")])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_composite_bridge_graph_built_exact_roundtrip(
    heteroatom, carbon_orders, prefix, unsaturation, endpoints, reverse
):
    source, _, _ = _bridged_quinoline(heteroatom, carbon_orders, endpoints)
    if reverse:
        source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    mol = read_rdkit_mol(source)
    plan = plan_bridged_fusion_wrapper(mol, frozenset(mol.atoms), mode=FusionMode.GENERAL)

    assert plan is not None
    expected = f"{','.join(endpoints)}-({prefix})quinoline"
    assert plan.rendered_name == expected
    (bridge,) = plan.bridges
    assert bridge.kind is NondetachableBridgeKind.COMPOSITE
    assert tuple(mol.atoms[atom].symbol for atom in bridge.atom_ids) == (heteroatom,) + ("C",) * (
        len(carbon_orders) + 1
    )
    assert bridge.endpoint_locants == endpoints
    assert bridge.internal_bond_orders == (1, *carbon_orders)
    assert bridge.unsaturation_locants == unsaturation
    assert not plan.bridge_unsaturation_operations
    assert plan.saturated_bridge_precursor is None
    assert plan.audit_checks[-1] == "complete_wrapper_graph_reconstruction"
    numbered_path = bridge.atom_ids if endpoints[0] == "5" else tuple(reversed(bridge.atom_ids))
    assert [plan.atom_to_locant[atom] for atom in numbered_path] == [str(i) for i in range(9, 9 + len(numbered_path))]
    result = name_mol(source)
    assert result.name == expected
    check = verify_with_opsin(result.name, Chem.MolToSmiles(source), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("endpoints,carbon_locant", [(("5", "2"), "10"), (("2", "5"), "9")])
@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_composite_bridge_carbonyl_uses_completed_system_map(endpoints, carbon_locant):
    source, path, _ = _bridged_quinoline("O", (), endpoints, oxo=True)
    mol = read_rdkit_mol(source)
    ring_atoms = frozenset(atom.GetIdx() for atom in source.GetAtoms() if atom.IsInRing())
    plan = plan_bridged_fusion_wrapper(mol, ring_atoms, mode=FusionMode.GENERAL)
    assert plan is not None
    assert plan.atom_to_locant[path[-1]] == carbon_locant
    result = name_mol(source, fusion_mode=FusionMode.GENERAL)
    assert result.name == f"{','.join(endpoints)}-(epoxymethano)quinolin-{carbon_locant}-one"
    check = verify_with_opsin(result.name, Chem.MolToSmiles(source), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize(
    "heteroatom,carbon_orders", [("S", ()), ("O", (1,)), ("N", ()), ("N", (1,)), ("O", (3,)), ("O", (2,))]
)
def test_unverified_composite_constructions_abstain(heteroatom, carbon_orders):
    source, path, atom_by_locant = _bridged_quinoline(heteroatom, carbon_orders)
    mol = read_rdkit_mol(source)
    locants = {atom: locant for locant, atom in atom_by_locant.items()}
    assert _bridge_class(mol, path) is None
    assert _bridge_operations(mol, (path,), frozenset(locants), locants) is None


def test_composite_annelated_edge_guard_is_preserved():
    source, path, atom_by_locant = _bridged_quinoline("O", (), ("3", "2"))
    mol = read_rdkit_mol(source)
    locants = {atom: locant for locant, atom in atom_by_locant.items()}
    assert _bridge_class(mol, path) is not None
    assert _bridge_operations(mol, (path,), frozenset(locants), locants) is None


def test_charged_composite_path_abstains():
    source, path, _ = _bridged_quinoline("O", ())
    source.GetAtomWithIdx(path[0]).SetFormalCharge(1)
    source.GetAtomWithIdx(path[0]).SetNumExplicitHs(1)
    Chem.SanitizeMol(source)
    mol = read_rdkit_mol(source)
    assert _bridge_class(mol, path) is None
    assert plan_bridged_fusion_wrapper(mol, frozenset(mol.atoms), mode=FusionMode.AUDITED_PIN) is None


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_composite_brackets_do_not_change_mixed_bridge_citation_order():
    source, _, atom_by_locant = _bridged_quinoline("O", ())
    editable = Chem.RWMol(source)
    oxygen = editable.AddAtom(Chem.Atom("O"))
    editable.AddBond(atom_by_locant["6"], oxygen, Chem.BondType.SINGLE)
    editable.AddBond(atom_by_locant["8"], oxygen, Chem.BondType.SINGLE)
    Chem.SanitizeMol(editable)
    result = name_mol(editable.GetMol())
    assert result.name == "6,8-epoxy-5,2-(epoxymethano)quinoline"
    check = verify_with_opsin(result.name, Chem.MolToSmiles(editable), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("corruption", ["prefix", "direction", "bond_ownership"])
def test_composite_bridge_audit_rejects_tampered_construction(corruption):
    source, _, _ = _bridged_quinoline("O", ())
    mol = read_rdkit_mol(source)
    plan = plan_bridged_fusion_wrapper(mol, frozenset(mol.atoms), mode=FusionMode.GENERAL)
    (bridge,) = plan.bridges
    if corruption == "prefix":
        bridge = replace(bridge, prefix="(methanoepoxy)")
    elif corruption == "direction":
        bridge = replace(
            bridge,
            atom_ids=tuple(reversed(bridge.atom_ids)),
            endpoint_atom_ids=tuple(reversed(bridge.endpoint_atom_ids)),
            endpoint_locants=tuple(reversed(bridge.endpoint_locants)),
        )
    else:
        bridge = replace(bridge, bond_ids=frozenset())
    assert (
        _audit_bridge_plan(
            mol,
            frozenset(mol.atoms),
            plan.parent.atom_ids,
            dict(plan.parent.locant_maps[0]),
            (bridge,),
            plan.parent.selected_bond_model,
            plan.derivative_state,
            retained_parent_redistribution=True,
        )
        is None
    )


@pytest.mark.parametrize("corruption", ["schema", "duplicate", "prefix", "orders", "heteroatom"])
def test_composite_construction_data_rejects_malformed_records(corruption):
    from copy import deepcopy

    data = deepcopy(load_json_table("fusion_composite_bridges.json"))
    if corruption == "schema":
        data["schema_version"] = 2
    elif corruption == "duplicate":
        data["constructions"].append(data["constructions"][0])
    elif corruption == "prefix":
        data["constructions"][0]["hetero_prefix"] = ""
    elif corruption == "orders":
        data["constructions"][0]["carbon_bond_orders"] = [True]
    else:
        data["constructions"][0]["heteroatom"] = "C"
    with pytest.raises(ValueError):
        composite_bridge_constructions_from_data(data)
