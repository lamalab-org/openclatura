import pytest

from openclatura import name_smiles
from openclatura.molecule import Molecule
from openclatura.retained_monocycle_templates import match_retained_monocycle_templates


def _cycle_graph(
    symbols: tuple[str, ...],
    *,
    double_edges: set[frozenset[int]],
    atom_ids: tuple[int, ...] | None = None,
) -> Molecule:
    ids = atom_ids or tuple(range(len(symbols)))
    mol = Molecule()
    for atom_id, symbol in zip(ids, symbols, strict=True):
        mol.add_atom(symbol, idx=atom_id)
    for index, left in enumerate(ids):
        right = ids[(index + 1) % len(ids)]
        order = 2 if frozenset((index, (index + 1) % len(ids))) in double_edges else 1
        mol.add_bond(left, right, order=order)
    return mol


@pytest.mark.parametrize(
    ("symbols", "double_edges", "expected"),
    [
        (("Se", "C", "C", "C", "C"), {frozenset((1, 2)), frozenset((3, 4))}, "selenophene"),
        (("Te", "C", "C", "C", "C"), {frozenset((1, 2)), frozenset((3, 4))}, "tellurophene"),
        (("P", "C", "C", "C", "C"), {frozenset((1, 2)), frozenset((3, 4))}, "phosphole"),
        (("As", "C", "C", "C", "C"), {frozenset((1, 2)), frozenset((3, 4))}, "arsole"),
        (("B", "C", "C", "C", "C"), {frozenset((1, 2)), frozenset((3, 4))}, "borole"),
        (("N", "N", "N", "N", "N"), {frozenset((1, 2)), frozenset((3, 4))}, "pentazole"),
        (
            ("N", "N", "C", "N", "N", "C"),
            {frozenset((0, 5)), frozenset((1, 2)), frozenset((3, 4))},
            "1,2,4,5-tetrazine",
        ),
        (
            ("P", "C", "C", "C", "C", "C"),
            {frozenset((0, 5)), frozenset((1, 2)), frozenset((3, 4))},
            "phosphinine",
        ),
        (
            ("B", "C", "C", "C", "C", "C"),
            {frozenset((0, 5)), frozenset((1, 2)), frozenset((3, 4))},
            "borinine",
        ),
    ],
)
def test_graph_templates_match_elements_and_locanted_bond_orders(symbols, double_edges, expected):
    mol = _cycle_graph(symbols, double_edges=double_edges)

    matches = match_retained_monocycle_templates(mol, set(mol.atoms))

    assert matches
    assert matches[0].template.name == expected


def test_graph_template_locants_do_not_depend_on_atom_ids():
    atom_ids = (40, 3, 17, 8, 25)
    mol = _cycle_graph(
        ("Se", "C", "C", "C", "C"),
        double_edges={frozenset((1, 2)), frozenset((3, 4))},
        atom_ids=atom_ids,
    )

    match = match_retained_monocycle_templates(mol, set(atom_ids))[0]

    assert any(mapping[40] == "1" for mapping in match.atom_to_locant_maps)


def test_graph_template_rejects_same_counts_with_wrong_bond_positions():
    mol = _cycle_graph(
        ("Se", "C", "C", "C", "C"),
        double_edges={frozenset((0, 1)), frozenset((2, 3))},
    )

    assert match_retained_monocycle_templates(mol, set(mol.atoms)) == ()


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("c1cc[se]c1", "selenophene"),
        ("c1cc[te]c1", "tellurophene"),
        ("c1cc[pH]c1", "1H-phosphole"),
        ("C1=C[AsH]C=C1", "1H-arsole"),
        ("B1C=CC=C1", "borole"),
        ("n1nn[nH]n1", "pentazole"),
        ("c1nncnn1", "1,2,4,5-tetrazine"),
        ("c1ccpcc1", "phosphinine"),
        ("b1ccccc1", "borinine"),
    ],
)
def test_public_namer_uses_graph_backed_retained_monocycles(smiles, expected):
    assert name_smiles(smiles) == expected
