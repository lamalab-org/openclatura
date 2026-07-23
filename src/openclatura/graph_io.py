"""SMILES input and graph component helpers."""

from rdkit import Chem

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

    return _build_molecule(rdmol, atom_metadata)


def read_rdkit_mol(rdmol: Chem.Mol | None, *, copy: bool = True) -> Molecule:
    """Convert an existing RDKit molecule into the internal graph model.

    Naming needs a kekulized graph, so the input is copied by default and the
    caller's molecule is left untouched.  Pass ``copy=False`` only when the
    molecule is a throwaway.  Molecules that were never sanitized (for example
    ``MolFromSmiles(..., sanitize=False)`` or a partially read SD record) get
    the minimal property-cache and ring perception the namer relies on.

    Explicit hydrogens, as SD records usually carry them, are folded back into
    implicit hydrogen counts; that renumbers the heavy atoms of such a
    molecule, so atom indices in a trace refer to the hydrogen-suppressed
    graph rather than to the input molecule.
    """

    if rdmol is None:
        return Molecule()
    if copy:
        rdmol = Chem.Mol(rdmol)

    _ensure_perception(rdmol)
    if any(atom.GetAtomicNum() == 1 for atom in rdmol.GetAtoms()):
        try:
            rdmol = Chem.RemoveHs(rdmol, sanitize=False)
            _ensure_perception(rdmol)
        except Exception:
            pass

    atom_metadata = _atom_metadata(rdmol)
    try:
        Chem.Kekulize(rdmol, clearAromaticFlags=True)
    except Exception:
        pass

    return _build_molecule(rdmol, atom_metadata)


def _ensure_perception(rdmol: Chem.Mol) -> None:
    """Give an externally supplied molecule the valence and ring data we need."""

    try:
        rdmol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        rdmol.GetRingInfo().NumRings()
    except RuntimeError:
        Chem.FastFindRings(rdmol)


def _build_molecule(rdmol: Chem.Mol | None, atom_metadata: dict | None) -> Molecule:
    """Populate the internal graph model from a prepared RDKit molecule."""

    mol = Molecule()
    if rdmol is None:
        return mol
    if atom_metadata is None:
        atom_metadata = _atom_metadata(rdmol)

    Chem.AssignStereochemistry(rdmol, force=True, cleanIt=True)
    chiral_centers = dict(Chem.FindMolChiralCenters(rdmol, includeUnassigned=False))

    for atom in rdmol.GetAtoms():
        stereo = chiral_centers.get(atom.GetIdx())
        if stereo and atom.GetSymbol() == "S" and atom.GetTotalDegree() == 3:
            stereo = "R" if stereo == "S" else "S"
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
        stereo = None
        st = bond.GetStereo()
        if st == Chem.rdchem.BondStereo.STEREOE:
            stereo = "E"
        elif st == Chem.rdchem.BondStereo.STEREOZ:
            stereo = "Z"

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
