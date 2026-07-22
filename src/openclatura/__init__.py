"""openclatura — deterministic SMILES → IUPAC name generator."""

from collections.abc import Iterable
from typing import Any

from .describer import DescribedComponent, Description, DescriptionTokenSummary, describe
from .engine import DEFAULT_NAMING_ENGINE, NamingEngine, NamingRequest, NamingResult
from .functional_groups import register_group_detector
from .human_descriptor import HumanDescription, describe_human
from .molecule import (
    AtomBinding,
    BondBinding,
    DecisionTrace,
    FunctionalGroupMetadata,
    NameAnalysis,
    NomenclatureOperation,
    OperationClass,
    TracePhase,
    TraceStep,
)
from .namer import (
    analyze_rdkit_mol,
    analyze_smiles,
    name_rdkit_mol,
    name_rdkit_mol_with_trace,
    name_smiles,
    name_smiles_with_trace,
)
from .naming_context import NamingIntent
from .nomenclature import RULES, registry
from .opsin_verify import OpsinCheck, OpsinStatus, opsin_available, verify_with_opsin


def name(
    smiles: str,
    *,
    include_trace: bool = False,
    verify_opsin: bool = False,
    token_debug: bool = False,
) -> NamingResult:
    """One-shot naming with the default engine. Returns a typed ``NamingResult``.

    The bare-string ``name_smiles`` helper is preserved for backwards
    compatibility; new code should prefer ``name`` (or ``analyze``) which
    returns a structured result with rules-hit information and optional
    OPSIN verification metadata.
    """

    return DEFAULT_NAMING_ENGINE.run(
        NamingRequest(
            smiles=smiles,
            include_trace=include_trace,
            verify_opsin=verify_opsin,
            token_debug=token_debug,
        )
    )


def name_mol(
    rdkit_mol,
    *,
    include_trace: bool = False,
    verify_opsin: bool = False,
    token_debug: bool = False,
) -> NamingResult:
    """One-shot naming of an existing RDKit molecule. Returns a ``NamingResult``.

    The molecule-shaped counterpart of :func:`name`, for callers who already
    have an ``rdkit.Chem.rdchem.Mol`` (say from an SD file) and would rather
    not round-trip through SMILES.  The input molecule is left unmodified, and
    ``result.smiles`` is only populated when ``verify_opsin`` requires it.
    """

    return DEFAULT_NAMING_ENGINE.run(
        NamingRequest(
            rdkit_mol=rdkit_mol,
            include_trace=include_trace,
            verify_opsin=verify_opsin,
            token_debug=token_debug,
        )
    )


def name_many(
    smiles_iter: Iterable[str | Any],
    *,
    include_trace: bool = False,
    verify_opsin: bool = False,
    token_debug: bool = False,
    processes: int | None | str = 1,
    chunksize: int = 64,
) -> list[NamingResult]:
    """Batch convenience wrapper around :meth:`NamingEngine.name_many`.

    Items may be SMILES strings or RDKit molecules, in any mix.
    """

    if processes == "auto":
        processes = None

    return DEFAULT_NAMING_ENGINE.name_many(
        smiles_iter,
        include_trace=include_trace,
        verify_opsin=verify_opsin,
        token_debug=token_debug,
        processes=processes,
        chunksize=chunksize,
    )


__version__ = "0.1.5"

__all__ = [
    "AtomBinding",
    "BondBinding",
    "DecisionTrace",
    "DescribedComponent",
    "Description",
    "DescriptionTokenSummary",
    "FunctionalGroupMetadata",
    "HumanDescription",
    "NameAnalysis",
    "NamingEngine",
    "NamingIntent",
    "NamingRequest",
    "NamingResult",
    "NomenclatureOperation",
    "OperationClass",
    "OpsinCheck",
    "OpsinStatus",
    "RULES",
    "TracePhase",
    "TraceStep",
    "__version__",
    "analyze_rdkit_mol",
    "analyze_smiles",
    "describe",
    "describe_human",
    "name",
    "name_many",
    "name_mol",
    "name_rdkit_mol",
    "name_rdkit_mol_with_trace",
    "name_smiles",
    "name_smiles_with_trace",
    "opsin_available",
    "register_group_detector",
    "registry",
    "verify_with_opsin",
]
