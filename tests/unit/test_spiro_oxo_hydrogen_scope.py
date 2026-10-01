"""Carbonyl-induced saturation is added H, not a singleton hydro prefix."""

from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.assembly_spiro import _scope_spiro_oxo_hydrogens
from openclatura.opsin_verify import verify_with_opsin

SMILES = "O=C1C[C@H](c2cc(F)c(F)c(F)c2)c2c(ccc3c2OC2(CCCCC2)CC3=O)O1"


def _molecule(variant):
    mol = Chem.MolFromSmiles(SMILES)
    if variant == "defluorinated":
        editable = Chem.RWMol(mol)
        for atom in reversed(list(mol.GetAtoms())):
            if atom.GetAtomicNum() == 9:
                editable.RemoveAtom(atom.GetIdx())
        mol = editable.GetMol()
        Chem.SanitizeMol(mol)
    elif variant == "enantiomer":
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                atom.InvertChirality()
    return mol


@pytest.mark.parametrize("variant", ["pubchem-18668", "defluorinated", "enantiomer"])
def test_spiro_oxo_added_hydrogens_keep_metadata_and_atom_order(variant):
    mol = _molecule(variant)
    names = []
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(mol, order), include_trace=True, verify_self=True)
        assert result.error is None
        assert result.name.endswith("-4,8(3H,9H)-dione")
        assert "hydro" not in result.name
        assert result.self_audit.coverage is not None
        assert not result.self_audit.coverage.unnamed_atoms
        names.append(result.name)
    assert names[0] == names[1]


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("variant", ["pubchem-18668", "defluorinated", "enantiomer"])
def test_spiro_oxo_added_hydrogens_exact_opsin(variant):
    mol = _molecule(variant)
    result = name_mol(mol, include_trace=True, verify_self=True)
    assert result.error is None
    check = verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip


@pytest.fixture
def component_parts(monkeypatch):
    import openclatura.assembly_spiro as assembly

    captured = []

    def capture(parts, junctions):
        if parts.principal_group is not None and any(
            operation.operation_kind == "additive_hydrogen" for operation in parts.hydro_operations
        ):
            captured.append((deepcopy(parts), set(junctions)))
        _scope_spiro_oxo_hydrogens(parts, junctions)

    monkeypatch.setattr(assembly, "_scope_spiro_oxo_hydrogens", capture)
    result = name_mol(Chem.MolFromSmiles(SMILES), include_trace=True)
    assert result.error is None
    assert len(captured) == 1
    return captured[0]


def test_projection_preserves_audited_plan_and_nonhydro_bindings(component_parts):
    parts, junctions = component_parts
    original = deepcopy(parts)
    _scope_spiro_oxo_hydrogens(parts, junctions)
    assert parts.parent_hydride == original.parent_hydride
    assert parts.parent_bond_orders_by_locants == original.parent_bond_orders_by_locants
    assert parts.parent_atom_ids_by_locant == original.parent_atom_ids_by_locant
    assert parts.principal_group == original.principal_group
    assert [binding for binding in parts.name_atom_bindings if binding.stage != "hydro"] == [
        binding for binding in original.name_atom_bindings if binding.stage != "hydro"
    ]
    assert all(operation.key == "added_hydrogen" for operation in parts.hydro_operations)
    assert {locant for operation in parts.hydro_operations for locant in operation.locants} == {"3", "9"}
    assert set(parts.indicated_hydrogens) == {"3", "9"}
    assert {atom for operation in parts.hydro_operations for atom in operation.atom_ids} == {
        parts.parent_atom_ids_by_locant[locant] for locant in ("3", "9")
    }
    projected = deepcopy(parts)
    _scope_spiro_oxo_hydrogens(parts, junctions)
    assert parts == projected


@pytest.mark.parametrize(
    "missing_proof", ["suffix", "observed_single", "assigned_double", "numbering", "junction", "neutral"]
)
def test_single_hydro_requires_graph_and_suffix_evidence(component_parts, missing_proof):
    parts, junctions = component_parts
    if missing_proof == "suffix":
        parts.principal_group.locants.remove("8")
    elif missing_proof == "observed_single":
        parts.parent_bond_orders_by_locants[("8", "9")] = 2
    elif missing_proof == "assigned_double":
        parent = parts.parent_hydride
        plan = parent.fusion_plan
        state = plan.derivative_state
        delta = state.bond_delta
        edge = {parts.parent_atom_ids_by_locant[locant] for locant in ("8", "9")}
        assignment = replace(
            delta.assignment,
            orders=tuple((pair, 1 if set(pair) == edge else order) for pair, order in delta.assignment.orders),
        )
        parts.parent_hydride = replace(
            parent,
            fusion_plan=replace(
                plan, derivative_state=replace(state, bond_delta=replace(delta, assignment=assignment))
            ),
        )
    elif missing_proof == "numbering":
        parts.parent_atom_ids_by_locant["9"] = -1
    elif missing_proof == "junction":
        junctions.add("9")
    else:
        parts.parent_atom_charges_by_locant["9"] = -1
    original = deepcopy(parts)
    _scope_spiro_oxo_hydrogens(parts, junctions)
    assert parts == original


def test_even_hydrogenation_pair_is_not_reclassified(component_parts):
    parts, junctions = component_parts
    parts.hydro_operations = [
        replace(item, key="additive_hydrogen", operation_kind="additive_hydrogen") for item in parts.hydro_operations
    ]
    original = deepcopy(parts)
    _scope_spiro_oxo_hydrogens(parts, junctions)
    assert parts == original
