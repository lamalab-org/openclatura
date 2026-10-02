"""Focused integration tests for retained polycyclic fusion components."""

from __future__ import annotations

import tempfile

import pytest

from openclatura import opsin_available
from openclatura.fusion.descriptor import build_fusion_name_ast, render_fusion_name
from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_smiles


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
def test_opsin_derived_naphtho_azulene_uses_polycyclic_component_cover():
    import py2opsin

    expected = "naphtho[2,3-f]azulene"
    with tempfile.TemporaryDirectory() as tmpdir:
        smiles = py2opsin.py2opsin(expected, tmp_fpath=f"{tmpdir}/input.txt")

    mol = read_smiles(smiles)
    faces = select_bounded_face_model(mol, mol.atoms)
    assert faces is not None
    registry = fusion_component_registry()
    matches = registry.match_faces(mol, faces)

    ast = build_fusion_name_ast(mol, matches, registry)

    assert render_fusion_name(ast, registry) == expected
    assert {match.spec_key for match in ast.component_occurrences} == {"naphthalene", "azulene"}
    assert all(len(match.covered_face_ids) == 2 for match in ast.component_occurrences)
    assert set().union(*(match.covered_face_ids for match in ast.component_occurrences)) == set(range(len(faces.faces)))
