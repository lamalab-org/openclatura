"""Graph-proof tests for P-25.5 third-component fusion handling."""

from openclatura.fusion.model import FusionMode
from openclatura.fusion.third_component import (
    _carbon_skeleton,
    _corresponding_carbon_graph_is_exact,
    plan_third_component_fusion_parent,
)
from openclatura.graph_io import read_smiles
from openclatura.molecule import Molecule

THIRD_COMPONENT_PARENT = "N1=C2C3=C(N=NC3=N1)N=N2"


def _cactus_third_component_graph() -> Molecule:
    """Return five pentagons whose component graph has two cyclic blocks."""

    mol = Molecule()
    for atom_id in range(15):
        mol.add_atom(
            "N" if atom_id in {2, 4} else "C",
            idx=atom_id,
            is_aromatic=True,
        )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (3, 8),
        (8, 9),
        (7, 0),
        (9, 1),
        (6, 2),
        (4, 10),
        (10, 11),
        (11, 12),
        (12, 3),
        (5, 13),
        (13, 14),
        (14, 10),
    )
    for bond_id, (left, right) in enumerate(edges, start=1):
        mol.add_bond(left, right, idx=bond_id)
    return mol


def _cactus_with_pendant_face() -> Molecule:
    """Extend the cactus by a pentagon outside either cyclic cover block."""

    mol = _cactus_third_component_graph()
    for atom_id in (15, 16, 17):
        mol.add_atom("C", idx=atom_id, is_aromatic=True)
    next_bond = max(mol.bonds) + 1
    for offset, (left, right) in enumerate(((13, 15), (15, 16), (16, 17), (17, 14))):
        mol.add_bond(left, right, idx=next_bond + offset)
    return mol


def test_cyclic_component_cover_becomes_audited_skeletal_replacement_parent():
    mol = read_smiles(THIRD_COMPONENT_PARENT)

    plan = plan_third_component_fusion_parent(
        mol,
        mol.atoms,
        mode=FusionMode.GENERAL,
    )

    assert plan is not None
    assert plan.cover_topology == "unicyclic"
    assert plan.ring_sizes == (5, 5, 5)
    assert plan.prohibited_citation.plan_kind == "cyclic_component_cover"
    assert plan.prohibited_citation.citation_plan is not None
    assert plan.prohibited_citation.citation_plan.cycle_closing_join_indices
    assert plan.parent.is_skeletal_replacement_fusion
    assert plan.parent.audit_ok
    assert set(plan.replacement_atom_ids) == {atom_id for atom_id, atom in mol.atoms.items() if atom.symbol == "N"}
    assert all(set(locant_map) == set(mol.atoms) for locant_map in plan.parent.proof_locant_maps)


def test_corresponding_carbon_certificate_rejects_graph_corruption():
    mol = read_smiles(THIRD_COMPONENT_PARENT)
    atoms = frozenset(mol.atoms)
    replacements = tuple(sorted(atom_id for atom_id, atom in mol.atoms.items() if atom.symbol != "C"))
    carbon = _carbon_skeleton(mol, atoms)

    assert _corresponding_carbon_graph_is_exact(mol, carbon, atoms, replacements)

    first_bond = next(iter(carbon.bonds))
    carbon.update_bond(first_bond, order=1 if carbon.bonds[first_bond].order != 1 else 2)
    assert not _corresponding_carbon_graph_is_exact(mol, carbon, atoms, replacements)


def test_carbon_surrogate_owns_its_valence_hydrogens():
    mol = Molecule()
    for atom, symbol in enumerate(("N", "O", "C", "C", "C")):
        mol.add_atom(symbol, idx=atom)
    for left, right in ((0, 1), (0, 2), (0, 3), (1, 4)):
        mol.add_bond(left, right)
    atoms = frozenset(mol.atoms)
    carbon = _carbon_skeleton(mol, atoms)
    assert carbon.atoms[0].total_h_count == 1
    assert carbon.atoms[1].total_h_count == 2
    assert carbon.atoms[2].total_h_count == 3
    assert _corresponding_carbon_graph_is_exact(mol, carbon, atoms, (0, 1))
    carbon.update_atom(0, total_h_count=0)
    assert not _corresponding_carbon_graph_is_exact(mol, carbon, atoms, (0, 1))


def test_cactus_component_cover_uses_the_same_replacement_parent_route():
    mol = _cactus_third_component_graph()

    plan = plan_third_component_fusion_parent(
        mol,
        mol.atoms,
        mode=FusionMode.GENERAL,
    )

    assert plan is not None
    assert plan.cover_topology == "cactus"
    assert plan.ring_sizes == (5, 5, 5, 5, 5)
    assert plan.parent.is_skeletal_replacement_fusion
    assert plan.parent.audit_ok
    assert len(plan.prohibited_citation.citation_plan.cycle_closing_join_indices) == 2


def test_cactus_cover_with_a_pendant_component_uses_the_same_proof_route():
    mol = _cactus_with_pendant_face()

    plan = plan_third_component_fusion_parent(
        mol,
        mol.atoms,
        mode=FusionMode.GENERAL,
    )

    assert plan is not None
    assert plan.cover_topology == "cactus"
    assert plan.ring_sizes == (5, 5, 5, 5, 5, 5)
    assert plan.parent.is_skeletal_replacement_fusion
    assert plan.parent.audit_ok
    assert len(plan.prohibited_citation.citation_plan.cycle_closing_join_indices) == 2


def test_third_component_planning_is_cached_even_when_not_applicable(monkeypatch):
    import openclatura.fusion.third_component as module

    mol = read_smiles("c1ccccc1")
    calls = 0

    def not_applicable(*args, **kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(module, "_plan_uncached", not_applicable)

    assert module.plan_third_component_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL) is None
    assert module.plan_third_component_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL) is None
    assert calls == 1


def test_neutral_carbon_and_charged_parents_abstain_safely():
    carbon = read_smiles("C1=C2C3=C(C=CC3=C1)C=C2")
    assert (
        plan_third_component_fusion_parent(
            carbon,
            carbon.atoms,
            mode=FusionMode.GENERAL,
        )
        is None
    )

    charged = read_smiles(THIRD_COMPONENT_PARENT)
    nitrogen = next(atom_id for atom_id, atom in charged.atoms.items() if atom.symbol == "N")
    charged.update_atom(nitrogen, charge=1)
    assert (
        plan_third_component_fusion_parent(
            charged,
            charged.atoms,
            mode=FusionMode.GENERAL,
        )
        is None
    )
