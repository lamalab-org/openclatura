"""Acceptance specification for element-independent acyl/chalcogen handling.

Target elements: O, S, Se, Te, Po, varied independently at every site.
Reviewed API: upstream commit 4da31d1ec84dd6cc36747f1c9e2284902e05fdb1.

Copy this file into tests/unit/ of an Openclatura checkout and run:
    pytest tests/unit/test_all_chalcogen_acyl_regressions.py -q

This is a regression specification, NOT an implementation patch. Openclatura
was not installed in the review environment, so these assertions were NOT run.
The 1,050 generated SMILES (900 acid/anhydride assignments and 150 ester
controls) were separately parsed and sanitized with RDKit 2025.09.4; all had
zero formal charge and zero radical electrons. This checks representations,
not chemical stability, preferred names, or Openclatura support.

The existing perception whitelist rejects Po. Po cases deliberately remain
normal acceptance tests, not skipped/xfail cases. A production implementation
must supply validated nomenclature data as well as the generic topology logic;
bypassing a validation guard is not evidence of complete Po naming support.
These matrices primarily test active group recognition and role preservation.
They do not constitute preferred-name validation for all 1,050 structures.

The preserved O/S name examples apply IUPAC Blue Book P-65.1.5.1 and
P-65.7.1 through P-65.7.4:
    https://iupac.qmul.ac.uk/BlueBook/P6.html
No Po name has been invented for a golden-name assertion.
"""

import random
from itertools import product

import pytest
from rdkit import Chem

import openclatura as oc
from openclatura.chalcogen_roles import (
    Chalcogen,
    ChalcogenLigandRole,
    DerivativeKind,
    FunctionalFamily,
)
from openclatura.chalcogen_vocabulary import resolve_anhydride_rule
from openclatura.graph_io import read_smiles
from openclatura.perception import perceive_groups

# This is the requested target scope, deliberately independent of the current
# whitelist. Otherwise missing support can disappear from test enumeration.
ELEMENTS = ("O", "S", "Se", "Te", "Po")
BONDED = {element: f"[{element}]" for element in ELEMENTS}
TERMINAL_H = {element: f"[{element}H]" for element in ELEMENTS}

NAME_CASES = (
    ("CCC(=O)O", "propanoic acid"),
    ("CCC(=S)S", "propanedithioic acid"),
    ("CCC(=S)O", "propanethioic O-acid"),
    ("CCC(=O)S", "propanethioic S-acid"),
    ("CCC(=S)OC(=O)", "formic propanethioic anhydride"),
    ("CCC(=S)OSC(=S)CC", "propanethioic thioperoxyanhydride"),
    ("CCC(=S)OC(=S)CC", "propanethioic anhydride"),
    ("CC(=O)OOC(=O)C", "acetic peroxyanhydride"),
    ("CC(=O)SOC(=O)CC", "S-acetic O-propanoic thioperoxyanhydride"),
    ("CC(=O)OSC(=O)CC", "O-acetic S-propanoic thioperoxyanhydride"),
    ("CC(=O)OOC(=O)CC", "acetic propanoic peroxyanhydride"),
)

ACID_CASES = tuple(
    (double, terminal, f"CCC(={BONDED[double]}){TERMINAL_H[terminal]}")
    for double, terminal in product(ELEMENTS, repeat=2)
)

PEROXYACID_CASES = tuple(
    (double, first, terminal, f"CCC(={BONDED[double]}){BONDED[first]}{TERMINAL_H[terminal]}")
    for double, first, terminal in product(ELEMENTS, repeat=3)
)

ANHYDRIDE_CASES = tuple(
    (left, tuple(middle), right, f"CCC(={BONDED[left]}){''.join(BONDED[e] for e in middle)}C(={BONDED[right]})CC")
    for bridge_length in (1, 2)
    for left, *middle, right in product(ELEMENTS, repeat=bridge_length + 2)
)

ESTER_CASES = tuple(
    (double, tuple(linker), f"CCC(={BONDED[double]}){''.join(BONDED[e] for e in linker)}C")
    for linker_length in (1, 2)
    for double, *linker in product(ELEMENTS, repeat=linker_length + 1)
)

GENERATED_SMILES = tuple(
    case[-1] for cases in (ACID_CASES, PEROXYACID_CASES, ANHYDRIDE_CASES, ESTER_CASES) for case in cases
)


def _describe(groups):
    """Diagnostics without assuming a particular public trace serialization."""
    return [
        (
            group.key,
            group.is_principal_candidate,
            getattr(group.descriptor, "derivative", None),
            tuple(ligand.element.value for ligand in group.descriptor.ligands) if group.descriptor is not None else (),
        )
        for group in groups
    ]


def _principal_acyl(smiles, derivative):
    mol = read_smiles(smiles)
    groups = perceive_groups(mol)
    candidates = [
        group
        for group in groups
        if group.is_principal_candidate
        and group.descriptor is not None
        and group.descriptor.family is FunctionalFamily.ACYL
        and group.descriptor.derivative is derivative
    ]
    assert len(candidates) == 1, (smiles, _describe(groups))
    # Passive descriptors are not sufficient: the group must be eligible.
    assert candidates[0].key != FunctionalFamily.ACYL.value, _describe(groups)
    return mol, candidates[0].descriptor, groups


def _single_ligand(desc, role):
    ligands = desc.ligands_with_role(role)
    assert len(ligands) == 1, (role, desc)
    return ligands[0]


def _check_ligand_bonds(mol, desc):
    for ligand in desc.ligands:
        bond = mol.get_bond(ligand.center, ligand.atom)
        assert bond is not None, (ligand, desc)
        assert bond.idx == ligand.bond_id, (ligand, bond)
        assert mol.atoms[ligand.atom].symbol == ligand.element.value
        if ligand.role is ChalcogenLigandRole.DOUBLE_BONDED:
            assert bond.order == 2
        else:
            assert bond.order == 1


def test_target_scope_and_matrix_sizes():
    assert set(ELEMENTS) <= {element.value for element in Chalcogen}
    assert len(ACID_CASES) == 25
    assert len(PEROXYACID_CASES) == 125
    assert len(ANHYDRIDE_CASES) == 750
    assert len(ESTER_CASES) == 150
    assert len(GENERATED_SMILES) == 1050


@pytest.mark.parametrize("smiles", GENERATED_SMILES)
def test_generated_input_has_explicit_closed_shell_intent(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in mol.GetAtoms()), smiles
    assert all(atom.GetFormalCharge() == 0 for atom in mol.GetAtoms()), smiles


@pytest.mark.parametrize("smiles, expected", NAME_CASES)
def test_reported_name_regressions(smiles, expected):
    assert oc.name(smiles).name == expected


@pytest.mark.parametrize("double, terminal, smiles", ACID_CASES)
def test_all_acid_site_assignments_remain_principal_acids(double, terminal, smiles):
    mol, desc, groups = _principal_acyl(smiles, DerivativeKind.ACID)
    assert len(desc.centers) == 1
    assert len(desc.ligands) == 2
    assert _single_ligand(desc, ChalcogenLigandRole.DOUBLE_BONDED).element.value == double
    terminal_ligand = _single_ligand(desc, ChalcogenLigandRole.HYDROGEN_BEARING)
    assert terminal_ligand.element.value == terminal
    assert mol.atoms[terminal_ligand.atom].total_h_count == 1
    _check_ligand_bonds(mol, desc)
    assert not any(
        group.is_principal_candidate
        and group.descriptor is not None
        and group.descriptor.family is FunctionalFamily.HYDROXY
        for group in groups
    ), (smiles, _describe(groups))


@pytest.mark.parametrize("double, first, terminal, smiles", PEROXYACID_CASES)
def test_all_peroxyacid_assignments_keep_the_ordered_group(double, first, terminal, smiles):
    mol, desc, _groups = _principal_acyl(smiles, DerivativeKind.ACID)
    assert len(desc.centers) == 1
    assert len(desc.ligands) == 3
    assert _single_ligand(desc, ChalcogenLigandRole.DOUBLE_BONDED).element.value == double
    terminal_ligand = _single_ligand(desc, ChalcogenLigandRole.HYDROGEN_BEARING)
    assert terminal_ligand.element.value == terminal
    assert mol.atoms[terminal_ligand.atom].total_h_count == 1
    assert len(desc.linker_paths) == 1
    path = desc.linker_paths[0]
    assert len(path) == 3 and path[0] == desc.centers[0]
    assert tuple(mol.atoms[idx].symbol for idx in path[1:]) == (first, terminal)
    _check_ligand_bonds(mol, desc)


@pytest.mark.parametrize("left, bridge, right, smiles", ANHYDRIDE_CASES)
def test_all_anhydride_assignments_keep_both_acyl_ends(left, bridge, right, smiles):
    mol, desc, groups = _principal_acyl(smiles, DerivativeKind.ANHYDRIDE)
    assert len(desc.centers) == 2
    assert len(desc.linker_paths) == 1
    path = desc.linker_paths[0]
    assert len(path) == len(bridge) + 2
    assert {path[0], path[-1]} == set(desc.centers)
    doubles = desc.ligands_with_role(ChalcogenLigandRole.DOUBLE_BONDED)
    assert len(doubles) == 2
    by_center = {ligand.center: ligand.element.value for ligand in doubles}
    assert set(by_center) == set(desc.centers)
    # Preserve the FULL oriented signature, not just independent multisets:
    # left=Se, bridge=O-S, right=Te differs from left=Se, bridge=S-O, right=Te.
    observed = (
        by_center[path[0]],
        tuple(mol.atoms[idx].symbol for idx in path[1:-1]),
        by_center[path[-1]],
    )
    assert observed in ((left, bridge, right), (right, bridge[::-1], left)), (
        smiles,
        observed,
    )
    for a, b in zip(path, path[1:]):
        assert mol.get_bond(a, b).order == 1
    _check_ligand_bonds(mol, desc)
    assert not any(
        group.is_principal_candidate
        and group.descriptor is not None
        and group.descriptor.derivative is DerivativeKind.ESTER
        for group in groups
    ), (smiles, _describe(groups))


@pytest.mark.parametrize("double, linker, smiles", ESTER_CASES)
def test_all_ester_controls_stay_single_center_derivatives(double, linker, smiles):
    mol, desc, groups = _principal_acyl(smiles, DerivativeKind.ESTER)
    assert len(desc.centers) == 1
    assert len(desc.ligands) == len(linker) + 1
    assert _single_ligand(desc, ChalcogenLigandRole.DOUBLE_BONDED).element.value == double
    assert len(desc.linker_paths) == 1
    path = desc.linker_paths[0]
    assert path[0] == desc.centers[0]
    # One-link descriptors include the organic attachment atom in their path;
    # two-link descriptors currently end at the terminal chalcogen atom.
    path_elements = tuple(mol.atoms[idx].symbol for idx in path[1:] if mol.atoms[idx].symbol in ELEMENTS)
    assert path_elements == linker
    _check_ligand_bonds(mol, desc)
    assert not any(
        group.descriptor is not None and group.descriptor.derivative is DerivativeKind.ANHYDRIDE for group in groups
    ), (smiles, _describe(groups))


@pytest.mark.parametrize("left, bridge, right, smiles", ANHYDRIDE_CASES)
def test_standard_anhydride_alias_requires_all_three_oxygen_sites(left, bridge, right, smiles):
    # Resolver test uses even a passive descriptor so it can expose the
    # independent ligands[-1] indexing defect before projection is removed.
    groups = perceive_groups(read_smiles(smiles))
    descriptors = [
        group.descriptor
        for group in groups
        if group.descriptor is not None and group.descriptor.derivative is DerivativeKind.ANHYDRIDE
    ]
    assert descriptors, (smiles, _describe(groups))
    key, rule = resolve_anhydride_rule(descriptors[0])
    is_ordinary_alias = key == "anhydride" and rule is None
    assert is_ordinary_alias == (left == right == "O" and bridge == ("O",)), smiles


@pytest.mark.parametrize("smiles, expected", NAME_CASES)
def test_golden_names_are_invariant_under_atom_renumbering(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    rng = random.Random(125)
    permutations = [list(reversed(range(mol.GetNumAtoms())))]
    for _ in range(3):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        permutations.append(order)
    for order in permutations:
        alternate = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        assert oc.name(alternate).name == expected, (smiles, alternate)
