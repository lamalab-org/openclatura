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

from openclatura import name_mol
from openclatura.graph_io import read_rdkit_mol
from openclatura.retained_graph_templates import (
    RetainedGraphTemplate,
    match_retained_graph_templates,
    retained_graph_templates,
    template_molecule,
    validate_retained_graph_family_partition,
)

try:
    import py2opsin
except Exception:  # pragma: no cover - optional test dependency
    py2opsin = None


AUDITED_TEMPLATES = tuple(template for template in retained_graph_templates() if template.derivative_audit_enabled)

_SIDECHAIN_SHAPES = ("methyl", "ethyl", "propyl", "propan-2-yl", "butan-2-yl")


def test_retained_graph_provider_families_have_disjoint_topology_indexes():
    validate_retained_graph_family_partition()


@pytest.mark.parametrize(
    "template",
    tuple(template for template in AUDITED_TEMPLATES if template.charge_policy == "exact"),
    ids=lambda template: template.name,
)
def test_exact_charge_templates_reject_changed_parent_charge_and_invalidate_cache(template):
    mol = template_molecule(template)
    atom_ids = set(mol.atoms)
    initial = match_retained_graph_templates(
        mol,
        atom_ids,
        allow_nonaromatic=True,
        families=frozenset({template.family}),
    )
    assert any(match.template.name == template.name for match in initial)

    atom_idx = min(atom_ids)
    mol.set_atom_charge(atom_idx, mol.atoms[atom_idx].charge + 1)
    changed = match_retained_graph_templates(
        mol,
        atom_ids,
        allow_nonaromatic=True,
        families=frozenset({template.family}),
    )
    assert all(match.template.name != template.name for match in changed)


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


def _template_fixture(template: RetainedGraphTemplate) -> ParentDerivativeFixture:
    molecule = Chem.MolFromSmiles(_opsin(template.output_name))
    assert molecule is not None
    internal = read_rdkit_mol(molecule)
    match = next(
        match
        for match in match_retained_graph_templates(
            internal,
            set(internal.atoms),
            allow_nonaromatic=True,
            families=frozenset({template.family}),
        )
        if match.template.name == template.name
    )
    assert match.template.derivative_production_enabled
    return ParentDerivativeFixture(template.name, molecule, match.locant_to_atom)


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


def _assert_retained_core_selected(template: RetainedGraphTemplate, fixture: ParentDerivativeFixture) -> None:
    for derivative in _random_derivatives(fixture):
        analysis = name_mol(derivative.molecule, include_trace=True, verify_opsin=True)
        assert analysis.opsin_check is not None
        assert analysis.opsin_check.status == "matched", (
            fixture.name,
            derivative.attachments,
            analysis.name,
            analysis.opsin_check,
        )
        retained = [step for step in analysis.decisions if step.decision == "used retained parent name"]
        assert retained, (fixture.name, derivative.attachments, analysis.name)
        assert retained[-1].data["retained_name"] == fixture.name

        selected = [step for step in analysis.decisions if step.decision == "selected parent skeleton"]
        assert selected, (fixture.name, derivative.attachments, analysis.name)
        assert set(selected[-1].atoms) == derivative.core_atom_ids
        if template.pre_descriptor_selection:
            assert selected[-1].data["polycycle_descriptor"] is None
            assert selected[-1].data["ring_parent_proof_source"] == "retained_template"
            assert selected[-1].data["ring_parent_audit_ok"] is True


@pytest.mark.opsin
@pytest.mark.parametrize("parent_name", ("tetracene", "benz[a]anthracene"))
def test_retained_pahs_expose_audited_template_proof(parent_name):
    template = next(template for template in AUDITED_TEMPLATES if template.name == parent_name)
    result = name_mol(_template_fixture(template).molecule, include_trace=True)
    selected = next(step for step in result.decisions if step.decision == "selected parent skeleton")
    assert selected.data["ring_parent_kind"] == "retained_polycycle"
    assert selected.data["ring_parent_proof_source"] == "retained_template"
    assert selected.data["ring_parent_audit_ok"] is True


@pytest.mark.opsin
@pytest.mark.parametrize("template", AUDITED_TEMPLATES, ids=lambda template: template.name)
def test_derivative_enabled_templates_round_trip_random_sidechains(template):
    _assert_retained_core_selected(template, _template_fixture(template))
