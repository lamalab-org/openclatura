"""Scoped stereochemistry helpers for small cyclic substituents."""

from rdkit import Chem

from .assembly_parts import AssemblyParts
from .molecule import Molecule


def scoped_small_ring_stereo_features(
    mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc
) -> list[tuple[str, str]]:
    """Return local R/S/r/s descriptors for unassigned small-ring stereo.

    RDKit intentionally leaves some substituted small-ring centers without CIP
    labels because the two ring paths are symmetry-related until the local
    substituent scope and numbering are known.  For substituent ring parents we
    can use the final locant map as the local symmetry breaker, assign a local
    CIP label, and render non-attachment centers as pseudoasymmetric
    lower-case descriptors.
    """

    # OPSIN currently treats mixed local R/S + lower-case r/s descriptor groups
    # inside substituent scopes as ordinary absolute stereochemistry and often
    # cannot attach the locants to the named fragment. Until a grammar-backed
    # relative-stereo renderer exists for these scopes, prefer the generic
    # cis/trans fallback in parent_pipeline._add_relative_ring_stereo.
    if parts.is_substituent:
        return []

    if not parts.is_substituent or not parts.is_ring or parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return []
    if not 3 <= len(numbered_path) <= 7:
        return []
    raw_atoms = [
        atom_idx
        for atom_idx in numbered_path
        if mol.atoms[atom_idx].raw_stereo in {"CW", "CCW"} and not mol.atoms[atom_idx].stereo
    ]
    if len(raw_atoms) != 2:
        return []
    locants = {atom_idx: str(get_loc(atom_idx)) for atom_idx in numbered_path}
    if any(not locants[atom_idx].isdigit() for atom_idx in numbered_path):
        return []

    local_cip = _assign_local_cip_with_locant_labels(mol, numbered_path, locants, raw_atoms)
    if set(local_cip) != set(raw_atoms):
        return []

    attachment_locant = str(parts.attachment_locant)
    features: list[tuple[str, str]] = []
    for atom_idx in sorted(raw_atoms, key=lambda idx: int(locants[idx])):
        locant = locants[atom_idx]
        descriptor = local_cip[atom_idx]
        if locant != attachment_locant:
            descriptor = descriptor.lower()
        features.append((locant, descriptor))
    return features


def _assign_local_cip_with_locant_labels(
    mol: Molecule, numbered_path: list[int], locants: dict[int, str], raw_atoms: list[int]
) -> dict[int, str]:
    rd_mol, atom_to_rd = _to_rdkit_mol(mol, numbered_path, locants)
    if rd_mol is None:
        return {}
    Chem.AssignStereochemistry(rd_mol, force=True, cleanIt=True)
    result: dict[int, str] = {}
    rd_to_atom = {rd_idx: atom_idx for atom_idx, rd_idx in atom_to_rd.items()}
    raw_set = set(raw_atoms)
    for rd_atom in rd_mol.GetAtoms():
        atom_idx = rd_to_atom[rd_atom.GetIdx()]
        if atom_idx in raw_set and rd_atom.HasProp("_CIPCode"):
            result[atom_idx] = rd_atom.GetProp("_CIPCode")
    return result


def _to_rdkit_mol(
    mol: Molecule, numbered_path: list[int], locants: dict[int, str]
) -> tuple[Chem.Mol, dict[int, int]] | tuple[None, dict[int, int]]:
    editable = Chem.RWMol()
    atom_to_rd: dict[int, int] = {}
    parent_set = set(numbered_path)
    for atom_idx in sorted(mol.atoms):
        atom = mol.atoms[atom_idx]
        rd_atom = Chem.Atom(atom.symbol)
        rd_atom.SetFormalCharge(atom.charge)
        if atom_idx in parent_set:
            rd_atom.SetIsotope(100 + int(locants[atom_idx]))
        if atom.raw_stereo == "CW":
            rd_atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
        elif atom.raw_stereo == "CCW":
            rd_atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
        atom_to_rd[atom_idx] = editable.AddAtom(rd_atom)

    bond_types = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
    }
    for bond in sorted(mol.bonds.values(), key=lambda item: item.idx):
        bond_type = bond_types.get(bond.order)
        if bond_type is None:
            return None, {}
        editable.AddBond(atom_to_rd[bond.u], atom_to_rd[bond.v], bond_type)

    rd_mol = editable.GetMol()
    try:
        rd_mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(rd_mol)
    except Exception:
        return None, {}
    return rd_mol, atom_to_rd
