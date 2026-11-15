"""Graph comparison helpers for accepted resonance representations."""

from __future__ import annotations


def canonical_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES, or ``None`` when parsing fails."""

    if not smiles:
        return None
    try:
        from rdkit.Chem import CanonSmiles  # type: ignore
    except Exception:  # pragma: no cover - rdkit is a runtime dependency
        return None
    try:
        return CanonSmiles(smiles)
    except Exception:
        return None


def equivalent_smiles(smiles_a: str, smiles_b: str) -> bool:
    """Return whether two SMILES are equal under supported resonance classes."""

    canonical_a = canonical_smiles(smiles_a)
    canonical_b = canonical_smiles(smiles_b)
    if canonical_a is None or canonical_b is None:
        return False
    if canonical_a == canonical_b:
        return True
    normalized_a = _cyclopentadienide_resonance_canonical(smiles_a)
    normalized_b = _cyclopentadienide_resonance_canonical(smiles_b)
    if normalized_a is None or normalized_b is None:
        return False
    if normalized_a == normalized_b:
        return True
    normalized_a = _sulfur_ylide_resonance_canonical(normalized_a)
    normalized_b = _sulfur_ylide_resonance_canonical(normalized_b)
    if normalized_a is None or normalized_b is None:
        return False
    normalized_a = _terminal_chalcogen_resonance_canonical(normalized_a)
    normalized_b = _terminal_chalcogen_resonance_canonical(normalized_b)
    if normalized_a is None or normalized_b is None:
        return False
    return normalized_a == normalized_b or _kekule_resonance_equivalent(normalized_a, normalized_b)


def _kekule_resonance_equivalent(left: str, right: str) -> bool:
    """Recognize bounded pi rearrangements without moving H, charge or stereo.

    Bridged conjugated rings are not always marked aromatic by RDKit. Their
    alternate Kekule drawings can therefore have different canonical SMILES.
    Resonance generation, unlike tautomer enumeration, preserves the nuclei.
    Explicit per-atom checks additionally reject charge-separated contributors.
    """

    from collections import Counter

    from rdkit import Chem

    def atom_state(atom):
        return (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            atom.GetChiralTag(),
        )

    first, second = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
    if first is None or second is None or first.GetNumBonds() != second.GetNumBonds():
        return False
    states = tuple(atom_state(atom) for atom in first.GetAtoms())
    # Chiral tag parity depends on atom ordering; compare it only within the
    # generated graph, using canonical isomeric SMILES across the two inputs.
    if Counter(state[:-1] for state in states) != Counter(atom_state(atom)[:-1] for atom in second.GetAtoms()):
        return False
    target = Chem.MolToSmiles(second)
    for candidate in Chem.ResonanceMolSupplier(first, Chem.KEKULE_ALL, maxStructs=64):
        if tuple(atom_state(atom) for atom in candidate.GetAtoms()) != states:
            continue
        # Re-sanitize each drawing so aromaticity and canonical atom ranking
        # use the same rules as the supplied target graph.
        rendered = canonical_smiles(Chem.MolToSmiles(candidate))
        if rendered == target:
            return True
    return False


def _cyclopentadienide_resonance_canonical(smiles: str) -> str | None:
    """Put the charge of each cyclopentadienide ring on a canonical ring atom.

    The five carbons of an aromatic cyclopentadienide share one negative
    charge, so a substituted ring can be drawn with the charge on any of them
    (the metallocene parents are read back with it on locant 1).  Only an
    isolated all-carbon five-membered aromatic ring holding exactly one
    negative charge is touched; the hydrogens stay where they are.
    """

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    rings = []
    for ring in ring_info.AtomRings():
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if (
            len(ring) == 5
            and all(atom.GetAtomicNum() == 6 and atom.GetIsAromatic() for atom in atoms)
            and all(ring_info.NumAtomRings(idx) == 1 for idx in ring)
            and sorted(atom.GetFormalCharge() for atom in atoms) == [-1, 0, 0, 0, 0]
        ):
            rings.append(ring)
    if not rings:
        return canonical_smiles(smiles)

    # Pin every ring hydrogen before the charge moves, then rank the atoms on a
    # copy with the ring charge removed so the ranks do not depend on where the
    # charge was drawn.
    for ring in rings:
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            hydrogens = atom.GetTotalNumHs()
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(hydrogens)
    uncharged = Chem.Mol(mol)
    for ring in rings:
        for idx in ring:
            uncharged.GetAtomWithIdx(idx).SetFormalCharge(0)
    uncharged.UpdatePropertyCache(strict=False)
    ranks = list(Chem.CanonicalRankAtoms(uncharged, breakTies=False))

    for ring in rings:
        target = min(ring, key=lambda idx: (ranks[idx], idx))
        for idx in ring:
            mol.GetAtomWithIdx(idx).SetFormalCharge(-1 if idx == target else 0)
    try:
        return canonical_smiles(Chem.MolToSmiles(mol))
    except Exception:
        return None


def _terminal_chalcogen_resonance_canonical(smiles: str) -> str | None:
    """Compare balanced P(+)-O/S(-) and S(+)-O(-) with their double-bond forms.

    Only terminal, hydrogen-free anions can consume the complete central
    positive charge. No protons, radicals, remote charges or tautomers move.
    """

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    changed = False
    for center in mol.GetAtoms():
        symbol = center.GetSymbol()
        charge = center.GetFormalCharge()
        if symbol not in {"P", "S"} or charge not in {1, 2} or center.GetNumRadicalElectrons():
            continue
        if center.GetIsAromatic():
            continue
        ligands = []
        allowed_ligands = {"O", "S"} if symbol == "P" else {"O"}
        for bond in center.GetBonds():
            ligand = bond.GetOtherAtom(center)
            if (
                bond.GetBondType() == Chem.BondType.SINGLE
                and ligand.GetSymbol() in allowed_ligands
                and ligand.GetFormalCharge() == -1
                and ligand.GetDegree() == 1
                and ligand.GetTotalNumHs() == 0
                and ligand.GetNumRadicalElectrons() == 0
            ):
                ligands.append((bond.GetIdx(), ligand.GetIdx()))
        if len(ligands) != charge:
            continue
        hydrogens = center.GetTotalNumHs()
        neutral_valence = sum(bond.GetBondTypeAsDouble() for bond in center.GetBonds()) + hydrogens + charge
        if neutral_valence not in ({5} if symbol == "P" else {4, 6}):
            continue
        target = editable.GetAtomWithIdx(center.GetIdx())
        target.SetFormalCharge(0)
        target.SetNumExplicitHs(hydrogens)
        target.SetNoImplicit(True)
        for bond_idx, ligand_idx in ligands:
            editable.GetBondWithIdx(bond_idx).SetBondType(Chem.BondType.DOUBLE)
            ligand = editable.GetAtomWithIdx(ligand_idx)
            ligand.SetFormalCharge(0)
            ligand.SetNoImplicit(True)
        changed = True
    if not changed:
        return canonical_smiles(smiles)
    try:
        Chem.SanitizeMol(editable)
        return Chem.MolToSmiles(editable, canonical=True)
    except Exception:
        return None


def _sulfur_ylide_resonance_canonical(smiles: str) -> str | None:
    """Normalize R2S(+)-C(-), R2S(+)=C(-), and R2S=C ylide drawings."""

    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import rdchem  # type: ignore
    except Exception:  # pragma: no cover - rdkit is a runtime dependency
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw_mol = Chem.RWMol(mol)
    changed = False
    for sulfur in list(rw_mol.GetAtoms()):
        if sulfur.GetSymbol() != "S":
            continue
        match = _sulfur_ylide_bond(sulfur)
        if match is None:
            continue
        bond_idx, carbon_idx = match
        carbon_ligands = [
            bond.GetOtherAtomIdx(sulfur.GetIdx())
            for bond in sulfur.GetBonds()
            if bond.GetIdx() != bond_idx and bond.GetOtherAtom(sulfur).GetSymbol() == "C"
        ]
        if len(carbon_ligands) != 2:
            continue
        normalized_sulfur = rw_mol.GetAtomWithIdx(sulfur.GetIdx())
        normalized_carbon = rw_mol.GetAtomWithIdx(carbon_idx)
        ylide_bond = rw_mol.GetBondWithIdx(bond_idx)
        normalized_sulfur.SetFormalCharge(0)
        normalized_sulfur.SetNumExplicitHs(0)
        normalized_sulfur.SetNoImplicit(True)
        normalized_carbon.SetFormalCharge(0)
        normalized_carbon.SetNumExplicitHs(0)
        normalized_carbon.SetNumRadicalElectrons(0)
        normalized_carbon.SetNoImplicit(False)
        ylide_bond.SetBondType(rdchem.BondType.DOUBLE)
        changed = True
    if not changed:
        return canonical_smiles(smiles)
    normalized = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(normalized)
        return Chem.MolToSmiles(normalized, canonical=True)
    except Exception:
        return None


def _sulfur_ylide_bond(sulfur) -> tuple[int, int] | None:
    candidates = []
    for bond in sulfur.GetBonds():
        carbon = bond.GetOtherAtom(sulfur)
        if carbon.GetSymbol() != "C":
            continue
        order = int(bond.GetBondTypeAsDouble())
        charge_separated = sulfur.GetFormalCharge() > 0 and carbon.GetFormalCharge() < 0
        neutral_resonance = sulfur.GetFormalCharge() == 0 and carbon.GetFormalCharge() == 0
        if (order == 1 and charge_separated) or (order == 2 and (charge_separated or neutral_resonance)):
            candidates.append((bond.GetIdx(), carbon.GetIdx()))
    return candidates[0] if len(candidates) == 1 else None


__all__ = ["canonical_smiles", "equivalent_smiles"]
