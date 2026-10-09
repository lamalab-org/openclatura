"""The main row of a three-port seven-membered ring (FR-5.1.1).

FR-5.1.1 draws the main row through the two attachments that sit as far apart
as possible around the central ring. The third attachment is then off the row
and uppermost, so FR-5.3 starts the numbering there. Which ring that is follows
from where the shared edges fall on the central cycle, never from how large the
attached rings are.

A seven-membered ring has no pair of parallel edges, so its polygon centre sits
off the line joining the two row neighbours. The direction grid keeps a
neighbour on the same row only when the offset is exactly horizontal, so that
off-axis centre used to cost the row an entire ring: every drawing reported a
maximum horizontal row of two where three was available. The orientation
criteria then tied drawings that start in *different* terminal rings, and the
ordered locant criteria of P-25.3.3.1.2 chose among candidates that should
never have been admitted. Low heteroatom locants break a genuine surviving tie;
they do not make an inadmissible drawing admissible.

With the row drawn, each system keeps exactly the two traversals of its one
uppermost ring, and criterion (a) decides between those. For 45274 that is the
mirror pair N-2/N-3 and N-2 wins. For 43346 the uppermost ring is the middle
benzene, leaving N-5,6,7 against N-10,11,12; the triazole start that used to
win with N-1,2,3 is not produced by the maximal-row construction at all.

OPSIN numbers both of these from the other surviving alternative. As in
test_fusion_three_ring_orientation, what is pinned here is the admissible
candidate set and the published criterion that ranks it, not agreement with the
parser; each divergence is a relabelling of the same ring system.
"""

import pytest
from rdkit import Chem

from openclatura import name_smiles, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.faces import cached_bounded_face_model, typed_face_model
from openclatura.fusion.layout import _direction_grid_centers, preferred_intrinsic_layouts
from openclatura.fusion.numbering import (
    _fusion_atoms,
    _numbering_from_layout,
    completed_system_numbering_selection,
)
from openclatura.graph_io import read_rdkit_mol

CASES = (
    pytest.param(
        "CN1CCC2c3cc(Cl)ccc3Cc3ccccc3C2C1",
        (("2",), ("3",)),
        ("2",),
        id="45274-dibenzocycloheptapyridine",
    ),
    pytest.param(
        "O=c1c2ccccc2c2ccccc2c2c1nnn2-c1ccc([N+](=O)[O-])cc1",
        (("10", "11", "12"), ("5", "6", "7")),
        ("5", "6", "7"),
        id="43346-dibenzocycloheptatriazole",
    ),
)


def _model(smiles):
    mol = read_rdkit_mol(Chem.MolFromSmiles(smiles))
    atoms = frozenset(max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms)
    bounded = cached_bounded_face_model(mol, atoms)
    return mol, atoms, bounded, typed_face_model(mol, bounded)


def _heteroatom_locants(mol, atoms, numbering):
    locants = {atom: f"{locant.base}{locant.fusion_suffix}" for atom, locant in numbering.atom_to_locant}
    return tuple(sorted(locants[atom] for atom in atoms if mol.atoms[atom].symbol != "C"))


@pytest.mark.parametrize("smiles,admissible,selected", CASES)
def test_three_port_seven_ring_draws_the_three_ring_main_row(smiles, admissible, selected):
    _, _, _, face_model = _model(smiles)
    adjacent = frozenset(frozenset((left, right)) for left, right, _ in face_model.face_adjacency)
    layouts = preferred_intrinsic_layouts(face_model)
    assert layouts
    widest = 0
    for layout in layouts:
        centers = {face: (x, y) for face, x, y in layout.face_positions}
        grid = _direction_grid_centers(centers, adjacent) or centers
        rows: dict[int, int] = {}
        for _, (_, y) in grid.items():
            rows[y] = rows.get(y, 0) + 1
        widest = max(widest, max(rows.values()))
    assert widest == 3, "FR-5.1.1 places the central ring and its two widest-apart attachments in one row"
    # -row_count is the second component of the orientation score.
    assert layouts[0].orientation_score[1] == -3


@pytest.mark.parametrize("smiles,admissible,selected", CASES)
def test_only_the_uppermost_ring_starts_an_admitted_numbering(smiles, admissible, selected):
    mol, atoms, bounded, face_model = _model(smiles)
    layouts = preferred_intrinsic_layouts(face_model)
    fusion_atoms = _fusion_atoms(bounded)
    perimeter_cache: dict = {}
    produced = set()
    for index, layout in enumerate(layouts):
        numbering = _numbering_from_layout(
            mol, bounded, face_model, layout, index, fusion_atoms, perimeter_cache=perimeter_cache
        )
        if numbering is not None:
            produced.add(_heteroatom_locants(mol, atoms, numbering))
    assert produced == set(admissible)


@pytest.mark.parametrize("smiles,admissible,selected", CASES)
def test_low_heteroatom_locants_decide_the_surviving_tie(smiles, admissible, selected):
    mol, atoms, bounded, face_model = _model(smiles)
    selection = completed_system_numbering_selection(
        mol, bounded, face_model=face_model, layouts=preferred_intrinsic_layouts(face_model)
    )
    assert {_heteroatom_locants(mol, atoms, numbering) for numbering in selection.accepted} == {selected}


# Three attachments on an eight-membered ring, 2, 2 and 4 edges apart: the
# diametral benzo pair forms the main row and the triazole stands above it, so
# the numbering starts in the triazole. The octagon's mirror axis runs through
# two vertices, so even its diametral edges are not parallel and its centre
# missed the row exactly as a seven-membered ring's does. Until the row was
# drawn, no layout produced a map OPSIN admits.
EIGHT_RING_NAMES = (
    pytest.param(
        "CCCCc1ccc2c(c1)CCc1cc(OC(C)C)ccc1-c1nnn(C(C)(C)C)c1-2",
        "11-butyl-1-(tert-butyl)-6-isopropoxy-8,9-dihydrodibenzo[1',2':1,2;1'',2'':5,6]cycloocta[3,4-d][1,2,3]triazole",
        id="54539-dibenzocyclooctatriazole",
    ),
    # Reached through the bridged path, so it has no intrinsic layout of its
    # own; the row its unbridged framework now draws is what moves the locants.
    pytest.param(
        "COc1cccc2c1C=Cc1c(OC)cccc1C1=C2C2C=CC1O2",
        "8,11-dimethoxy-1,4-dihydro-1,4-epoxytribenzo[a,c,e][8]annulene",
        id="98874-tribenzoannulene",
    ),
)


def test_eight_membered_centre_also_draws_its_row():
    _, _, _, face_model = _model("CCCCc1ccc2c(c1)CCc1cc(OC(C)C)ccc1-c1nnn(C(C)(C)C)c1-2")
    layouts = preferred_intrinsic_layouts(face_model)
    assert layouts
    # -row_count is the second component of the orientation score.
    assert layouts[0].orientation_score[1] == -3


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles,expected", EIGHT_RING_NAMES)
def test_the_drawn_row_carries_the_eight_ring_locants(smiles, expected):
    produced = name_smiles(smiles)
    assert produced == expected
    assert verify_with_opsin(produced, smiles).status == "matched"
