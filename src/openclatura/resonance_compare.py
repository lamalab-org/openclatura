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
    return _sulfur_ylide_resonance_canonical(smiles_a) == _sulfur_ylide_resonance_canonical(smiles_b)


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
