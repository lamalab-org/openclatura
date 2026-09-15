"""Graph-built third-ring probes for inherited benzoheterocycle components."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol
from openclatura.retained_fused_templates import retained_graph_templates

CASES = (
    ("isoquinoline", None),
    ("cinnoline", None),
    ("phthalazine", None),
    ("quinazoline", None),
    ("quinoxaline", None),
    ("1H-indole", "1"),
    ("2H-isoindole", "2"),
    ("2-benzofuran", None),
    ("indazole", "1"),
    ("2-benzothiophene", None),
    ("1H-benzotriazole", "1"),
    ("2,1,3-benzoxadiazole", None),
    ("2,1,3-benzothiadiazole", None),
    ("1,4-benzodioxine", None),
    ("1H-1,5-benzodiazepine", "1"),
)


def _third_ring_derivative(key, hydrogen_locant, *, reverse=False):
    template = next(template for template in retained_graph_templates() if template.name == key)
    mol = Chem.RWMol()
    ids = {}
    for atom in template.atoms:
        rd_atom = Chem.Atom(atom.symbol)
        rd_atom.SetIsAromatic(True)
        if atom.locant == hydrogen_locant:
            rd_atom.SetNumExplicitHs(1)
        ids[atom.locant] = mol.AddAtom(rd_atom)
    for bond in template.bonds:
        mol.AddBond(*(ids[locant] for locant in bond.locants), Chem.BondType.AROMATIC)
    symbols = {atom.locant: atom.symbol for atom in template.atoms}
    benzene = next(ring for ring in template.rings if len(ring) == 6 and all(symbols[a] == "C" for a in ring))
    interfaces = [
        (left, right)
        for left, right in zip(benzene, benzene[1:] + benzene[:1])
        if left not in template.fusion_atoms and right not in template.fusion_atoms
    ]
    interface = interfaces[1]
    # Distinguish the third ring from a pyridazine already in the component,
    # which would otherwise prefer multiparent dipyridazine nomenclature.
    if key in {"cinnoline", "phthalazine"}:
        host, side, symbols = "[1,2,4]triazine", "e", ("N", "N", "C", "N")
    else:
        host, side, symbols = "pyridazine", "d", ("C", "N", "N", "C")
    path = [ids[interface[1]]]
    for symbol in symbols:
        atom = Chem.Atom(symbol)
        atom.SetIsAromatic(True)
        path.append(mol.AddAtom(atom))
    path.append(ids[interface[0]])
    for left, right in zip(path, path[1:]):
        mol.AddBond(left, right, Chem.BondType.AROMATIC)
    Chem.SanitizeMol(mol)
    if reverse:
        order = list(reversed(range(mol.GetNumAtoms())))
        mol = Chem.RenumberAtoms(mol, order)
        inverse = {old: new for new, old in enumerate(order)}
        ids = {locant: inverse[atom] for locant, atom in ids.items()}
    return mol, template, ids, f"[{','.join(interface)}-{side}]{host}"


@pytest.mark.parametrize("key,hydrogen_locant", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("reverse", (False, True), ids=("original", "reversed"))
def test_auto_component_matches_inherited_locants_in_third_ring_graph(key, hydrogen_locant, reverse):
    registry = fusion_component_registry()
    component = registry.get(key)
    assert component is not None, f"missing auto-registered component: {key}"
    assert component.spec.attached_prefix
    assert component.spec.usable_as_attached
    assert not component.spec.usable_as_parent
    rd_mol, template, ids, _ = _third_ring_derivative(key, hydrogen_locant, reverse=reverse)
    mol = read_rdkit_mol(rd_mol)
    faces = select_bounded_face_model(mol, mol.atoms)
    assert faces is not None and len(faces.faces) == 3

    matches = [
        match
        for match in registry.match_faces(mol, faces, role="attached")
        if match.spec_key == key and dict(match.local_to_input_atom) == ids
    ]

    assert matches, f"inherited locants were not preserved for {key}"
    match = matches[0]
    assert len(match.covered_face_ids) == 2
    spec = registry.spec_for_match(match)
    assert spec.template.name == template.name
    assert spec.template.atoms == template.atoms
    assert {frozenset(bond.locants) for bond in spec.bonds} == {frozenset(bond.locants) for bond in template.bonds}
    for atom in spec.atoms:
        assert mol.atoms[ids[atom.locant]].symbol == atom.symbol
    for bond in spec.bonds:
        left, right = (ids[locant] for locant in bond.locants)
        assert right in mol.get_neighbors(left)


@pytest.mark.opsin
@pytest.mark.parametrize("key,hydrogen_locant", CASES, ids=[case[0] for case in CASES])
def test_auto_component_attached_prefix_roundtrips_inherited_interface(key, hydrogen_locant):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    component = fusion_component_registry().get(key)
    assert component is not None
    mol, _, _, citation_suffix = _third_ring_derivative(key, hydrogen_locant)
    # Probe this exact inherited interface independently of alternative
    # covers selected by parent seniority in the full planner.
    citation = component.spec.attached_prefix + citation_suffix
    if hydrogen_locant:
        internal = read_rdkit_mol(mol)
        planned = plan_fusion_parent(internal, internal.atoms, mode=FusionMode.GENERAL)
        assert isinstance(planned, FusionConfirmed), planned
        citation = ",".join(f"{locant}H" for locant in planned.plan.indicated_hydrogens) + "-" + citation

    check = verify_with_opsin(citation, Chem.MolToSmiles(mol), standardize_smiles=False)

    assert check.status == "matched", check.to_dict()


@pytest.mark.opsin
@pytest.mark.parametrize("key,hydrogen_locant", CASES, ids=[case[0] for case in CASES])
def test_auto_component_third_ring_generated_name_passes_audit_and_opsin(key, hydrogen_locant):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    registry = fusion_component_registry()
    assert registry.get(key) is not None
    rd_mol, _, _, _ = _third_ring_derivative(key, hydrogen_locant)
    mol = read_rdkit_mol(rd_mol)

    planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(planned, FusionConfirmed), planned
    assert planned.plan.audit.confirmed
    # A seven-membered heteroring can become the parent of an alternative
    # phthalazino cover. It must still exercise an auto-registered prefix.
    selected_auto = [
        match
        for match in planned.plan.ast.component_occurrences
        if match.spec_key in {case[0] for case in CASES}
        and match.occurrence_id not in planned.plan.ast.parent_occurrences
    ]
    assert selected_auto
    for match in selected_auto:
        assert registry.spec_for_match(match).attached_prefix + "[" in planned.plan.rendered_base_name
    result = name_mol(rd_mol, fusion_mode=FusionMode.GENERAL, include_trace=True)
    assert result.error is None
    assert result.parent_nomenclature == "systematic_fusion"
    check = verify_with_opsin(result.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
