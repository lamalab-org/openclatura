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

# The third column is whether the component may be a base component. The first
# eight are ordinary retained components of Table 2.8 and are admitted as
# parents by P-25.3.2.1.3, so they carry an explicit registry row and the
# generic family never claims them. The last seven are the specially formed
# benzoheterocycles of P-25.2.2.4: they can be parents too, but only when the
# P-25.3.5 grouping conditions hold, and those checks do not exist yet, so the
# family policy still withholds the role.
CASES = (
    ("isoquinoline", None, True),
    ("cinnoline", None, True),
    ("phthalazine", None, True),
    ("quinazoline", None, True),
    ("quinoxaline", None, True),
    ("1H-indole", "1", True),
    ("2H-isoindole", "2", True),
    ("indazole", "1", True),
    ("2-benzofuran", None, False),
    ("2-benzothiophene", None, False),
    ("1H-benzotriazole", "1", False),
    ("2,1,3-benzoxadiazole", None, False),
    ("2,1,3-benzothiadiazole", None, False),
    ("1,4-benzodioxine", None, False),
    ("1H-1,5-benzodiazepine", "1", False),
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


@pytest.mark.parametrize("key,hydrogen_locant,base_eligible", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("reverse", (False, True), ids=("original", "reversed"))
def test_auto_component_matches_inherited_locants_in_third_ring_graph(key, hydrogen_locant, base_eligible, reverse):
    registry = fusion_component_registry()
    component = registry.get(key)
    assert component is not None, f"missing auto-registered component: {key}"
    assert component.spec.attached_prefix
    assert component.spec.usable_as_attached
    assert component.spec.usable_as_parent is base_eligible
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
@pytest.mark.parametrize("key,hydrogen_locant,base_eligible", CASES, ids=[case[0] for case in CASES])
def test_auto_component_attached_prefix_roundtrips_inherited_interface(key, hydrogen_locant, base_eligible):
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
@pytest.mark.parametrize("key,hydrogen_locant,base_eligible", CASES, ids=[case[0] for case in CASES])
def test_auto_component_third_ring_generated_name_passes_audit_and_opsin(key, hydrogen_locant, base_eligible):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    registry = fusion_component_registry()
    assert registry.get(key) is not None
    rd_mol, _, _, _ = _third_ring_derivative(key, hydrogen_locant)
    mol = read_rdkit_mol(rd_mol)

    planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(planned, FusionConfirmed), planned
    assert planned.plan.audit.confirmed
    # Which component covers this graph is a seniority decision, not a
    # registration one, and it moved once the retained components of Tables 2.7
    # and 2.8 became available: the third ring these probes add carries two ring
    # nitrogens, so the phthalazine cover now outranks the probed component on
    # heteroatom count and the probe is named pyrido[3,4-g]phthalazine rather
    # than on isoquinoline. That the component is registered with the right
    # inherited locants and prefix is what the two tests above check; this one
    # checks the plan the registry admits still audits and names.
    assert {match.spec_key for match in planned.plan.ast.component_occurrences}
    result = name_mol(rd_mol, fusion_mode=FusionMode.GENERAL, include_trace=True)
    assert result.error is None
    assert result.parent_nomenclature == "systematic_fusion"
    check = verify_with_opsin(result.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
