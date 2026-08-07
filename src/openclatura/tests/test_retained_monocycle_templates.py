import pytest

from openclatura import name_smiles
from openclatura.graph_io import read_smiles
from openclatura.molecule import Molecule
from openclatura.retained_fused_templates import match_retained_fused_templates
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


@pytest.mark.parametrize(
    ("symbols", "double_edges", "expected", "additive", "indicated"),
    [
        (("O", "C", "C", "C", "C"), {frozenset((3, 4))}, "furan", ("2", "3"), ()),
        (("O", "C", "C", "C", "C"), {frozenset((2, 3))}, "furan", ("2", "5"), ()),
        (("N", "C", "C", "C", "C"), {frozenset((2, 3))}, "pyrrole", ("2", "5"), ("1",)),
        (
            ("O", "C", "N", "C", "C"),
            {frozenset((1, 2))},
            "1,3-oxazole",
            ("4", "5"),
            (),
        ),
        (
            ("O", "C", "C", "C", "C", "C"),
            {frozenset((2, 3)), frozenset((4, 5))},
            "pyran",
            (),
            ("2",),
        ),
        (
            ("O", "C", "C", "C", "C", "C"),
            {frozenset((4, 5))},
            "pyran",
            ("3", "4"),
            ("2",),
        ),
        (
            ("N", "C", "C", "C", "C", "C"),
            {frozenset((3, 4))},
            "pyridine",
            ("1", "2", "3", "6"),
            (),
        ),
    ],
)
def test_family_states_carry_graph_bound_hydrogen_operations(
    symbols, double_edges, expected, additive, indicated
):
    atom_ids = tuple(100 + 7 * index for index in range(len(symbols)))
    mol = _cycle_graph(symbols, double_edges=double_edges, atom_ids=atom_ids)

    match = match_retained_monocycle_templates(mol, set(atom_ids))[0]

    assert match.template.name == expected
    assert match.metadata.additive_hydrogen_locants == additive
    assert match.metadata.indicated_hydrogen_locants == indicated
    assert all(set(mapping) == set(atom_ids) for mapping in match.atom_to_locant_maps)


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


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C1CSCO1", "1,3-oxathiolane"),
        ("C1CSCS1", "1,3-dithiolane"),
        ("C1=CCNC1", "2,5-dihydro-1H-pyrrole"),
        ("C1=COCC1", "2,3-dihydrofuran"),
        ("C1=CCOC1", "2,5-dihydrofuran"),
        ("C1=NCCO1", "4,5-dihydro-1,3-oxazole"),
        ("C1=NCCS1", "4,5-dihydro-1,3-thiazole"),
        ("C1=CCOC=C1", "2H-pyran"),
        ("C1=COC=CC1", "4H-pyran"),
        ("C1=CCSC=C1", "2H-thiopyran"),
        ("C1=CSC=CC1", "4H-thiopyran"),
        ("C1CSCCS1", "1,4-dithiane"),
        ("C1CNCOC1", "1,3-oxazinane"),
        ("C1CNCNC1", "hexahydropyrimidine"),
        ("C1NCNCN1", "1,3,5-triazinane"),
        ("C1=COCCC1", "3,4-dihydro-2H-pyran"),
        ("C1=CCNCC1", "1,2,3,6-tetrahydropyridine"),
        ("C1=CNCCC1", "1,2,3,4-tetrahydropyridine"),
    ],
)
def test_public_namer_uses_retained_monocycle_family_states(smiles, expected):
    assert name_smiles(smiles) == expected


def test_bare_fused_template_keeps_all_isoindole_locant_maps():
    mol = read_smiles("C1=NCc2ccccc21")

    matches = [
        match
        for match in match_retained_fused_templates(mol, set(mol.atoms), include_disabled=True)
        if match.template.name == "isoindole"
    ]

    assert matches
    assert {match.atom_to_locant[2] for match in matches} == {"1", "3"}


def test_bare_fused_numbering_is_invariant_to_isoindole_input_order():
    assert name_smiles("C1=NCc2ccccc21") == "1H-isoindole"
    assert name_smiles("C1N=CC2=CC=CC=C12") == "1H-isoindole"


def test_bare_fused_gate_does_not_enable_derivative_grammar():
    assert name_smiles("CC1=NCc2ccccc21") == "1-methyl-3H-isoindole"
