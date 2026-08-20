"""Derivative stability for every retained graph template added on this branch."""

from __future__ import annotations

import hashlib
import random
import shutil
import tempfile
import warnings
from dataclasses import dataclass

import pytest
from rdkit import Chem

from openclatura import analyze_rdkit_mol
from openclatura.graph_io import read_rdkit_mol
from openclatura.retained_fused_templates import (
    _generated_acene_templates,
    _generated_polyaphene_templates,
    match_retained_fused_templates,
)
from openclatura.retained_macrocycle_templates import match_retained_macrocycle

try:
    import py2opsin
except Exception:  # pragma: no cover - optional test dependency
    py2opsin = None


_EXPLICIT_BRANCH_TEMPLATE_NAMES = (
    "benz[a]anthracene",
    "benzo[a]pyrene",
    "benzo[b]fluoranthene",
    "benzo[e]pyrene",
    "benzo[ghi]perylene",
    "benzo[j]fluoranthene",
    "benzo[k]fluoranthene",
    "cyclopenta[cd]pyrene",
    "dibenz[a,h]anthracene",
    "indeno[1,2,3-cd]pyrene",
    "ovalene",
    "pyranthrene",
    "rubicene",
    "tetraphenylene",
    "trinaphthylene",
)

NEW_FUSED_TEMPLATE_NAMES = tuple(
    sorted(
        {
            *_EXPLICIT_BRANCH_TEMPLATE_NAMES,
            *(template.name for template in _generated_acene_templates()),
            *(template.name for template in _generated_polyaphene_templates()),
        }
    )
)

MACROCYCLE_SMILES = {
    "porphyrin": "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3",
    "corrin": "C1=C2CCC(=N2)C=C2CCC(N2)C2CCC(=N2)C=C2CCC1=N2",
}

_SIDECHAIN_SHAPES = ("methyl", "ethyl", "propyl", "propan-2-yl", "butan-2-yl")


@dataclass(frozen=True)
class ParentDerivativeFixture:
    name: str
    molecule: Chem.Mol
    locant_to_atom: dict[str, int]


@dataclass(frozen=True)
class RandomDerivative:
    molecule: Chem.Mol
    attachments: tuple[tuple[str, str], ...]
    core_atom_ids: frozenset[int]


def _require_opsin() -> None:
    if py2opsin is None:
        pytest.skip("py2opsin is not available")
    if shutil.which("java") is None:
        pytest.skip("Java runtime not found (OPSIN requires Java)")


def _opsin(name: str) -> str:
    _require_opsin()
    with tempfile.TemporaryDirectory(prefix="openclatura_derivative_opsin_") as tmpdir:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return py2opsin.py2opsin(name, tmp_fpath=f"{tmpdir}/input.txt")


def _fused_fixture(parent_name: str) -> ParentDerivativeFixture:
    molecule = Chem.MolFromSmiles(_opsin(parent_name))
    assert molecule is not None
    internal = read_rdkit_mol(molecule)
    match = next(
        match
        for match in match_retained_fused_templates(internal, set(internal.atoms), allow_nonaromatic=True)
        if match.template.name == parent_name
    )
    assert match.template.derivative_production_enabled
    assert match.template.pre_descriptor_selection
    return ParentDerivativeFixture(parent_name, molecule, match.locant_to_atom)


def _macrocycle_fixture(parent_name: str) -> ParentDerivativeFixture:
    molecule = Chem.MolFromSmiles(MACROCYCLE_SMILES[parent_name])
    assert molecule is not None
    internal = read_rdkit_mol(molecule)
    match = match_retained_macrocycle(internal, set(internal.atoms))
    assert match is not None
    return ParentDerivativeFixture(parent_name, molecule, match.locant_to_atom)


def _substitutable_carbon_locants(fixture: ParentDerivativeFixture) -> list[str]:
    return sorted(
        (
            locant
            for locant, atom_idx in fixture.locant_to_atom.items()
            if locant.isdigit()
            and fixture.molecule.GetAtomWithIdx(atom_idx).GetSymbol() == "C"
            and fixture.molecule.GetAtomWithIdx(atom_idx).GetTotalNumHs() > 0
        ),
        key=int,
    )


def _attach_sidechain(molecule: Chem.RWMol, parent_atom: int, shape: str) -> None:
    root = molecule.AddAtom(Chem.Atom("C"))
    molecule.AddBond(parent_atom, root, Chem.BondType.SINGLE)
    if shape == "methyl":
        return
    first = molecule.AddAtom(Chem.Atom("C"))
    molecule.AddBond(root, first, Chem.BondType.SINGLE)
    if shape == "ethyl":
        return
    if shape == "propan-2-yl":
        second = molecule.AddAtom(Chem.Atom("C"))
        molecule.AddBond(root, second, Chem.BondType.SINGLE)
        return
    second = molecule.AddAtom(Chem.Atom("C"))
    molecule.AddBond(first, second, Chem.BondType.SINGLE)
    if shape == "propyl":
        return
    branch = molecule.AddAtom(Chem.Atom("C"))
    molecule.AddBond(root, branch, Chem.BondType.SINGLE)


def _random_derivatives(fixture: ParentDerivativeFixture, count: int = 3):
    locants = _substitutable_carbon_locants(fixture)
    assert locants, fixture.name
    seed = int.from_bytes(hashlib.sha256(fixture.name.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    for _ in range(count):
        selected = rng.sample(locants, k=rng.randint(1, min(3, len(locants))))
        editable = Chem.RWMol(fixture.molecule)
        shapes = []
        for locant in selected:
            shape = rng.choice(_SIDECHAIN_SHAPES)
            _attach_sidechain(editable, fixture.locant_to_atom[locant], shape)
            shapes.append(shape)
        derivative = editable.GetMol()
        Chem.SanitizeMol(derivative)
        # Exercise graph matching independently of parser atom order.  RDKit's
        # permutation maps each new atom index to its old index.
        permutation = list(range(derivative.GetNumAtoms()))
        rng.shuffle(permutation)
        derivative = Chem.RenumberAtoms(derivative, permutation)
        core_atom_ids = frozenset(
            new_idx for new_idx, old_idx in enumerate(permutation) if old_idx < fixture.molecule.GetNumAtoms()
        )
        yield RandomDerivative(
            molecule=derivative,
            attachments=tuple(zip(selected, shapes, strict=True)),
            core_atom_ids=core_atom_ids,
        )


def _assert_retained_core_selected(fixture: ParentDerivativeFixture) -> None:
    for derivative in _random_derivatives(fixture):
        analysis = analyze_rdkit_mol(derivative.molecule)
        retained = [step for step in analysis.decisions if step.decision == "used retained parent name"]
        assert retained, (fixture.name, derivative.attachments, analysis.name)
        assert retained[-1].data["retained_name"] == fixture.name

        selected = [step for step in analysis.decisions if step.decision == "selected parent skeleton"]
        assert selected, (fixture.name, derivative.attachments, analysis.name)
        assert set(selected[-1].atoms) == derivative.core_atom_ids
        assert selected[-1].data["polycycle_descriptor"] is None


@pytest.mark.opsin
@pytest.mark.parametrize("parent_name", NEW_FUSED_TEMPLATE_NAMES)
def test_new_fused_templates_keep_retained_parent_for_random_sidechains(parent_name):
    _assert_retained_core_selected(_fused_fixture(parent_name))


@pytest.mark.parametrize("parent_name", tuple(MACROCYCLE_SMILES))
def test_new_macrocycle_templates_keep_retained_parent_for_random_sidechains(parent_name):
    _assert_retained_core_selected(_macrocycle_fixture(parent_name))
