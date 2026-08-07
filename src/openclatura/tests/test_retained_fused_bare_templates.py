import pytest

from openclatura import name_smiles
from openclatura.molecule import Molecule
from openclatura.retained_fused_templates import (
    match_retained_fused_template,
    retained_fused_graph_templates,
    template_molecule,
)

BARE_TEMPLATE_NAMES = (
    "2-benzofuran",
    "2-benzothiophene",
    "1,3-benzoxazole",
    "1,3-benzothiazole",
    "1H-benzotriazole",
    "2,1,3-benzoxadiazole",
    "2,1,3-benzothiadiazole",
    "2H-quinolizine",
    "1H-pyrrolizine",
    "2,3-dihydro-1-benzofuran",
    "2H-1-benzopyran",
    "4H-1-benzopyran",
    "3,4-dihydro-2H-1-benzopyran",
    "1,2,3,4-tetrahydroquinoline",
    "1,2,3,4-tetrahydroisoquinoline",
    "2,3-dihydro-1,4-benzodioxine",
    "9,10-dihydroacridine",
    "phenanthridine",
    "dibenzofuran",
    "dibenzothiophene",
    "10H-phenothiazine",
    "10H-phenoxazine",
    "7H-pyrrolo[2,3-d]pyrimidine",
    "1H-pyrazolo[3,4-d]pyrimidine",
    "naphtho[2,3-b]furan",
    "naphtho[1,2-b]thiophene",
    "naphtho[2,3-d][1,3]oxazole",
)


def _template(name):
    return next(
        template
        for template in retained_fused_graph_templates(include_disabled=True)
        if template.name == name and template.bare_parent_enabled
    )


def _remap_atom_ids(source: Molecule) -> Molecule:
    remapped = Molecule()
    atom_map = {atom_id: 1000 + 17 * offset for offset, atom_id in enumerate(reversed(tuple(source.atoms)))}
    for atom_id, atom in source.atoms.items():
        remapped.add_atom(
            atom.symbol,
            idx=atom_map[atom_id],
            charge=atom.charge,
            is_aromatic=atom.is_aromatic,
            explicit_h_count=atom.explicit_h_count,
            total_h_count=atom.total_h_count,
        )
    for bond in source.bonds.values():
        remapped.add_bond(atom_map[bond.u], atom_map[bond.v], order=bond.order)
    return remapped


@pytest.mark.parametrize("name", BARE_TEMPLATE_NAMES)
def test_bare_retained_fused_templates_match_remapped_graphs(name):
    template = _template(name)
    mol = _remap_atom_ids(template_molecule(template))

    match = match_retained_fused_template(mol, set(mol.atoms), template)

    assert match is not None
    assert set(match.atom_to_locant) == set(mol.atoms)
    assert set(match.atom_to_locant.values()) == set(template.locants)


def test_strict_fused_state_does_not_match_different_hydrogenation():
    aromatic = _template("2-benzofuran")
    hydrogenated = _template("2,3-dihydro-1-benzofuran")
    aromatic_graph = template_molecule(aromatic)
    hydrogenated_graph = template_molecule(hydrogenated)

    assert match_retained_fused_template(aromatic_graph, set(aromatic_graph.atoms), hydrogenated) is None
    assert match_retained_fused_template(hydrogenated_graph, set(hydrogenated_graph.atoms), aromatic) is None


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1ccc2cocc2c1", "2-benzofuran"),
        ("c1ccc2[nH]nnc2c1", "1H-benzotriazole"),
        ("c1ccc2c(c1)CCO2", "2,3-dihydro-1-benzofuran"),
        ("c1ccc2c(c1)cnc1ccccc12", "phenanthridine"),
        ("c1ccc2c(c1)oc1ccccc12", "dibenzofuran"),
        ("c1ncc2cc[nH]c2n1", "7H-pyrrolo[2,3-d]pyrimidine"),
        ("c1ccc2cc3occc3cc2c1", "naphtho[2,3-b]furan"),
    ],
)
def test_public_namer_uses_bare_retained_fused_templates(smiles, expected):
    assert name_smiles(smiles) == expected
