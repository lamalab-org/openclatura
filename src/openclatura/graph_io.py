"""SMILES input and graph component helpers."""

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from .molecule import Molecule


def read_smiles(smiles: str) -> Molecule:
    """Parse a SMILES string into the internal graph model."""

    rdmol = Chem.MolFromSmiles(smiles)
    atom_metadata = None
    if rdmol is None:
        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
        if rdmol:
            rdmol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(rdmol)
    else:
        atom_metadata = _atom_metadata(rdmol)
        try:
            Chem.Kekulize(rdmol, clearAromaticFlags=True)
        except Exception:
            pass

    mol = Molecule()
    if rdmol is None:
        return mol
    if atom_metadata is None:
        atom_metadata = _atom_metadata(rdmol)

    Chem.AssignStereochemistry(rdmol, force=True, cleanIt=True)
    # The legacy labeler above is still needed to perceive stereo units, but
    # its CIP codes mis-rank ligands in deep-sphere comparisons (e.g.
    # ring-closure duplicate atoms) and invert 3-coordinate sulfur centers.
    # rdCIPLabeler overwrites the _CIPCode properties with correct labels.
    try:
        rdCIPLabeler.AssignCIPLabels(rdmol)
    except Exception:
        pass  # unsanitized input; keep the legacy labels
    chiral_centers = {
        atom.GetIdx(): atom.GetProp("_CIPCode")
        for atom in rdmol.GetAtoms()
        if atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode") in ("R", "S")
    }

    for atom in rdmol.GetAtoms():
        stereo = chiral_centers.get(atom.GetIdx())
        raw_stereo = _raw_tetrahedral_stereo(atom) if not stereo else None
        mol.add_atom(
            symbol=atom.GetSymbol(),
            idx=atom.GetIdx(),
            charge=atom.GetFormalCharge(),
            stereo=stereo,
            raw_stereo=raw_stereo,
            is_aromatic=atom_metadata[atom.GetIdx()]["is_aromatic"],
            explicit_h_count=atom_metadata[atom.GetIdx()]["explicit_h_count"],
            total_h_count=atom_metadata[atom.GetIdx()]["total_h_count"],
        )

    for bond in rdmol.GetBonds():
        # rdCIPLabeler rewrites bond enums to STEREOCIS/STEREOTRANS but stamps
        # the authoritative E/Z label as _CIPCode; fall back to the enums when
        # the labeler did not run.
        stereo = bond.GetPropsAsDict().get("_CIPCode")
        if stereo not in ("E", "Z"):
            st = bond.GetStereo()
            if st == Chem.rdchem.BondStereo.STEREOE:
                stereo = "E"
            elif st == Chem.rdchem.BondStereo.STEREOZ:
                stereo = "Z"
            else:
                stereo = None

        in_small_ring = any(bond.IsInRingSize(i) for i in range(3, 8))

        mol.add_bond(
            u=bond.GetBeginAtomIdx(),
            v=bond.GetEndAtomIdx(),
            order=int(bond.GetBondTypeAsDouble()),
            stereo=stereo,
            in_small_ring=in_small_ring,
        )
    return mol


def _raw_tetrahedral_stereo(atom: Chem.Atom) -> str | None:
    """Return raw tetrahedral chirality tags that do not have CIP assignment."""

    tag = atom.GetChiralTag()
    if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        return "CW"
    if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return "CCW"
    return None


def _atom_metadata(rdmol: Chem.Mol) -> dict[int, dict[str, int | bool]]:
    """Return atom metadata that may be lost when RDKit kekulizes a molecule."""

    return {
        atom.GetIdx(): {
            "is_aromatic": atom.GetIsAromatic(),
            "explicit_h_count": atom.GetNumExplicitHs(),
            "total_h_count": atom.GetTotalNumHs(),
        }
        for atom in rdmol.GetAtoms()
    }


def get_connected_components(mol: Molecule) -> list[set[int]]:
    """Split a molecule graph into disconnected naming components."""

    visited = set()
    components = []
    for atom in mol:
        if atom.idx not in visited:
            comp = set()
            q = [atom.idx]
            while q:
                curr = q.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    q.extend(mol.get_neighbors(curr))
            components.append(comp)
    return components
