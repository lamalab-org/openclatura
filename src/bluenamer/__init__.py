"""bluenamer — deterministic SMILES → IUPAC name generator."""

from collections.abc import Iterable

from .describer import Description, DescriptionFacts, describe
from .engine import DEFAULT_NAMING_ENGINE, NamingEngine, NamingRequest, NamingResult
from .functional_groups import register_group_detector
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
from .namer import analyze_smiles, name_smiles, name_smiles_with_trace
from .naming_context import NamingIntent
from .nomenclature import RULES, registry
from .opsin_verify import OpsinCheck, OpsinStatus, opsin_available, verify_with_opsin


def name(
    smiles: str,
    *,
    include_trace: bool = False,
    verify_opsin: bool = False,
) -> NamingResult:
    """One-shot naming with the default engine. Returns a typed ``NamingResult``.

    The bare-string ``name_smiles`` helper is preserved for backwards
    compatibility; new code should prefer ``name`` (or ``analyze``) which
    returns a structured result with rules-hit information and optional
    OPSIN verification metadata.
    """

    return DEFAULT_NAMING_ENGINE.run(
        NamingRequest(smiles=smiles, include_trace=include_trace, verify_opsin=verify_opsin)
    )


def name_many(
    smiles_iter: Iterable[str],
    *,
    include_trace: bool = False,
    verify_opsin: bool = False,
    processes: int | None | str = 1,
    chunksize: int = 64,
) -> list[NamingResult]:
    """Batch convenience wrapper around :meth:`NamingEngine.name_many`."""

    if processes == "auto":
        processes = None

    return DEFAULT_NAMING_ENGINE.name_many(
        smiles_iter,
        include_trace=include_trace,
        verify_opsin=verify_opsin,
        processes=processes,
        chunksize=chunksize,
    )


__version__ = "0.1.0"

__all__ = [
    "AtomBinding",
    "BondBinding",
    "DecisionTrace",
    "Description",
    "DescriptionFacts",
    "FunctionalGroupMetadata",
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
    "analyze_smiles",
    "describe",
    "name",
    "name_many",
    "name_smiles",
    "name_smiles_with_trace",
    "opsin_available",
    "register_group_detector",
    "registry",
    "verify_with_opsin",
]
