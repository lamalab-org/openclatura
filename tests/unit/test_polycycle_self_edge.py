"""Malformed legacy bridge candidates must not escape the topology audit."""

import pytest
from rdkit import Chem

from openclatura import chains, name_mol, name_smiles, opsin_available, verify_with_opsin
from openclatura.graph_io import read_smiles
from openclatura.molecule import Molecule
from openclatura.polycycle_topology import audit_von_baeyer_descriptor, build_von_baeyer_numbering

SMILES = "C=C(C)[C@H]1CC[C@@]23O[C@@H]2[C@@H](C/C(C)=C2/C(=O)C=C(C)[C@@]2(O)CC1)OC3=O"
RING_EDGES = {
    (3, 4),
    (3, 22),
    (4, 5),
    (5, 6),
    (6, 7),
    (6, 8),
    (6, 24),
    (7, 8),
    (8, 9),
    (9, 10),
    (9, 23),
    (10, 11),
    (11, 13),
    (13, 14),
    (13, 19),
    (14, 16),
    (16, 17),
    (17, 19),
    (19, 21),
    (21, 22),
    (23, 24),
}
BAD_PATH = (6, 24, 23, 9, 10, 11, 13, 14, 16, 17, 19, 21, 22, 3, 4, 5, 6, 7, 8)
BAD_DESCRIPTOR = "tetracyclo[15.2.0.0^{4,19}.0^{7,11}]"


def test_duplicate_path_is_rejected_before_edge_reconstruction(monkeypatch):
    def fail_reconstruction(*args):
        raise AssertionError("malformed path reached edge reconstruction")

    monkeypatch.setattr("openclatura.polycycle_topology._von_baeyer_edges_from_numbering", fail_reconstruction)
    audit = audit_von_baeyer_descriptor(BAD_DESCRIPTOR, BAD_PATH, RING_EDGES)
    numbering = build_von_baeyer_numbering(BAD_DESCRIPTOR, BAD_PATH, RING_EDGES)

    assert not audit.audit_ok
    assert not numbering.audit_ok
    assert "numbering path contains duplicate atoms" in audit.audit_errors
    assert not audit.expected_edges


@pytest.mark.parametrize(
    "descriptor,path",
    [
        ("tricyclo[2.2.1.0^{2,6}]", (1, 2, 3)),
        ("tricyclo[-1.2.1.0^{2,3}]", (1, 2, 3, 4)),
        ("tricyclo[2.2.1.0^{2,99}]", tuple(range(1, 8))),
        ("tricyclo[2.2.1.1^{2,8}]", tuple(range(1, 9))),
        ("tricyclo[2.2.1.0^{2,2}]", tuple(range(1, 8))),
    ],
)
def test_malformed_descriptors_fail_closed(descriptor, path):
    edges = {(a, b) for a, b in zip(path, path[1:])} | {(path[0], path[-1])}
    audit = audit_von_baeyer_descriptor(descriptor, path, edges)
    assert not audit.audit_ok
    assert audit.audit_errors
    assert all(a != b for a, b in audit.expected_edges)


@pytest.mark.parametrize("subdivide", [False, True])
@pytest.mark.parametrize("reverse_labels", [False, True])
def test_graph_variants_have_bijective_legacy_paths_and_audited_fallback(subdivide, reverse_labels):
    edges = set(RING_EDGES)
    if subdivide:
        edges.remove((3, 4))
        edges.update({(3, 30), (4, 30)})
    labels = {a: 100 - 3 * a if reverse_labels else a for edge in edges for a in edge}
    edges = {tuple(sorted((labels[a], labels[b]))) for a, b in edges}
    atoms = {a for edge in edges for a in edge}
    mol = Molecule()
    for a in sorted(atoms):
        mol.add_atom("C", idx=a)
    for a, b in sorted(edges):
        mol.add_bond(a, b)

    descriptor, paths = chains.get_von_baeyer_descriptor_and_path(atoms, edges)
    if descriptor:
        assert all(len(path) == len(set(path)) == len(atoms) for path in paths)
    candidate = chains._polyspiro_or_von_baeyer_candidate(mol, atoms, edges)
    assert candidate.descriptor_allowed
    assert candidate.numberings
    assert all(numbering.audit_ok for numbering in candidate.numberings)
    assert all(set(numbering.path) == atoms for numbering in candidate.numberings)
    valid_path = candidate.paths[0]
    malformed_path = [valid_path[0], *valid_path[:-1]]
    numberings = chains._audited_von_baeyer_numberings(
        mol, candidate.descriptor, [malformed_path, valid_path], frozenset(edges)
    )
    assert [numbering.path for numbering in numberings] == [tuple(valid_path)]


def test_index88584_fusion_is_attempted_before_descriptor_fallback(monkeypatch):
    mol = read_smiles(SMILES)
    attempts = []
    original_fusion = chains._confirmed_fusion_numbering_paths
    original_candidate = chains._polyspiro_or_von_baeyer_candidate

    def fusion(mol, atoms):
        attempts.append(frozenset(atoms))
        return original_fusion(mol, atoms)

    def candidate(mol, atoms, edges):
        assert frozenset(atoms) in attempts
        return original_candidate(mol, atoms, edges)

    monkeypatch.setattr(chains, "_confirmed_fusion_numbering_paths", fusion)
    monkeypatch.setattr(chains, "_polyspiro_or_von_baeyer_candidate", candidate)
    systems = chains.find_ring_systems(mol)
    assert attempts
    assert systems
    assert all(system.paths for system in systems)
    assert name_smiles(SMILES)


@pytest.mark.opsin
@pytest.mark.parametrize("variant", ["original", "reversed", "no_stereo", "ethyl"])
def test_index88584_public_name_exact_opsin_roundtrip(variant):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    smiles = SMILES
    mol = Chem.MolFromSmiles(smiles)
    if variant == "reversed":
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        smiles = Chem.MolToSmiles(mol, canonical=False)
    elif variant == "no_stereo":
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    elif variant == "ethyl":
        smiles = smiles.replace("C=C(C)", "C=C(CC)", 1)
    result = name_mol(Chem.MolFromSmiles(smiles), include_trace=True)
    assert result.error is None
    assert result.parent_nomenclature == "bridged_fusion"
    selected = next(step for step in result.decisions if step.decision == "selected audited bridged fusion parent")
    assert "complete_wrapper_graph_reconstruction" in selected.data["audit_checks"]
    assert [bridge["kind"] for bridge in selected.data["bridges"]] == ["composite"]
    name = result.name
    assert name == name_smiles(smiles)
    check = verify_with_opsin(name, smiles, standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip
