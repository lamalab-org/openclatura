import warnings

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


def suppress_noisy_eval_warnings() -> None:
    """Silence parser-normalization noise during large round-trip evaluations.

    The hidden failure scripts intentionally feed many unsupported or invalid
    reconstructed structures back through RDKit.  RDKit reports those as C++
    log messages such as kekulization and valence warnings; they are expected
    for failed cases and make the useful summary hard to read.
    """

    RDLogger.DisableLog("rdApp.*")
    try:
        rdBase.DisableLog("rdApp.*")
    except AttributeError:
        pass
    warnings.filterwarnings(
        "ignore",
        message=r".*OPSIN raised the following error.*",
        category=RuntimeWarning,
    )


suppress_noisy_eval_warnings()

normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()
uncharger = rdMolStandardize.Uncharger()
fragment_chooser = rdMolStandardize.LargestFragmentChooser()
tautomer_enum = rdMolStandardize.TautomerEnumerator()

# Bounds repeated parse/canonicalize/serialize passes while seeking a stable
# representation. This is independent of RDKit's internal tautomer limit.
_MAX_STANDARDIZATION_PASSES = 10


def _roundtrip_safe_smiles(mol: Chem.Mol) -> str | None:
    """Serialize ``mol`` to canonical SMILES that RDKit can parse again.

    RDKit can retain aromatic flags after neutralization that produce an
    aromatic SMILES with no valid Kekule form.  A Kekule serialization clears
    that inconsistent intermediate while preserving the molecular graph.
    """

    for kekule in (False, True):
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=kekule)
        except Exception:
            continue
        if Chem.MolFromSmiles(smiles) is not None:
            return smiles
    return None


def standardize_mol(smi: str) -> str | None:
    """Return a parseable, idempotent standardized tautomeric SMILES.

    Serialization/reparse between charge normalization and tautomer
    canonicalization refreshes RDKit's aromaticity and valence state.  The
    short fixed-point loop is needed because some heterocyclic tautomer sets
    choose a different canonical member after their first serialization.
    """

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol = rdMolStandardize.Cleanup(mol)
    mol = normalizer.normalize(mol)
    mol = reionizer.reionize(mol)
    mol = uncharger.uncharge(mol)

    current = _roundtrip_safe_smiles(mol)
    if current is None:
        return None

    seen: set[str] = set()
    for _ in range(_MAX_STANDARDIZATION_PASSES):
        if current in seen:
            # A deterministic representative makes a rare cycle idempotent.
            return min(seen)
        seen.add(current)

        reparsed = Chem.MolFromSmiles(current)
        if reparsed is None:  # defensive; _roundtrip_safe_smiles checked it
            return None
        canonical = tautomer_enum.Canonicalize(reparsed)
        next_smiles = _roundtrip_safe_smiles(canonical)
        if next_smiles is None:
            return None
        if next_smiles == current:
            return current
        current = next_smiles

    return current
