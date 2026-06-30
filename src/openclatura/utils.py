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


def standardize_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol = rdMolStandardize.Cleanup(mol)
    mol = normalizer.normalize(mol)
    mol = reionizer.reionize(mol)
    mol = uncharger.uncharge(mol)
    mol = tautomer_enum.Canonicalize(mol)

    return Chem.MolToSmiles(mol, canonical=True)
