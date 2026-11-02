"""openclatura — deterministic SMILES → IUPAC name generator."""

from collections.abc import Iterable
from typing import Any

from ._version import __version__
from .describer import DescribedComponent, Description, DescriptionTokenSummary, describe
from .engine import DEFAULT_NAMING_ENGINE, NamingEngine, NamingRequest, NamingResult
from .functional_groups import register_group_detector
from .fusion.model import FusionMode
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
from .naming_style import PRESETS as STYLE_PRESETS
from .naming_style import NamingStyle, resolve_style
from .nomenclature import RULES, registry
from .opsin_verify import OpsinCheck, OpsinStatus, opsin_available, verify_with_opsin


def name(
    smiles: str,
    *,
    include_trace: bool = False,
    verify_opsin: bool = False,
    verify_self: bool = False,
    token_debug: bool = False,
    omit_redundant_locants: bool = True,
    fusion_mode: FusionMode | str = FusionMode.AUDITED_PIN,
    style: NamingStyle | str | None = None,
    enclosing_marks: str | None = None,
    thiol_prefix: str | None = None,
    locant_placement: str | None = None,
    heterocycle_names: str | None = None,
) -> NamingResult:
    """One-shot naming with the default engine. Returns a typed ``NamingResult``.

    The bare-string ``name_smiles`` helper is preserved for backwards
    compatibility; new code should prefer ``name`` (or ``analyze``) which
    returns a structured result with rules-hit information and optional
    verification metadata.  ``verify_opsin`` round-trips the name through OPSIN
    (needs Java); ``verify_self`` runs the dependency-free reconstruction audit
    (see ``result.self_audit`` / ``result.self_verified``). Systematic fusion
    naming defaults to the audited PIN policy; pass an explicit ``FusionMode``
    to select general fusion nomenclature or legacy behavior.

    ``style`` picks a spelling preset (``"default"``, ``"iupac2013"``,
    ``"traditional"``; see ``STYLE_PRESETS``) or a ``NamingStyle``;
    ``enclosing_marks``, ``thiol_prefix``, ``locant_placement`` and
    ``heterocycle_names`` override single options of it.
    """

    return DEFAULT_NAMING_ENGINE.run(
        NamingRequest(
            smiles=smiles,
            include_trace=include_trace,
            verify_opsin=verify_opsin,
            verify_self=verify_self,
            token_debug=token_debug,
            omit_redundant_locants=omit_redundant_locants,
            fusion_mode=FusionMode(fusion_mode),
            style=resolve_style(
                style,
                enclosing_marks=enclosing_marks,
                thiol_prefix=thiol_prefix,
                locant_placement=locant_placement,
                heterocycle_names=heterocycle_names,
            ),
        )
    )


def name_mol(
    rdkit_mol,
    *,
    include_trace: bool = False,
    verify_opsin: bool = False,
    verify_self: bool = False,
    token_debug: bool = False,
    omit_redundant_locants: bool = True,
    fusion_mode: FusionMode | str = FusionMode.AUDITED_PIN,
    style: NamingStyle | str | None = None,
    enclosing_marks: str | None = None,
    thiol_prefix: str | None = None,
    locant_placement: str | None = None,
    heterocycle_names: str | None = None,
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
            verify_self=verify_self,
            token_debug=token_debug,
            omit_redundant_locants=omit_redundant_locants,
            fusion_mode=FusionMode(fusion_mode),
            style=resolve_style(
                style,
                enclosing_marks=enclosing_marks,
                thiol_prefix=thiol_prefix,
                locant_placement=locant_placement,
                heterocycle_names=heterocycle_names,
            ),
        )
    )


def name_many(
    smiles_iter: Iterable[str | Any],
    *,
    include_trace: bool = False,
    verify_opsin: bool = False,
    verify_self: bool = False,
    token_debug: bool = False,
    omit_redundant_locants: bool = True,
    fusion_mode: FusionMode | str = FusionMode.AUDITED_PIN,
    processes: int | None | str = 1,
    chunksize: int = 64,
    style: NamingStyle | str | None = None,
    enclosing_marks: str | None = None,
    thiol_prefix: str | None = None,
    locant_placement: str | None = None,
    heterocycle_names: str | None = None,
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
        verify_self=verify_self,
        token_debug=token_debug,
        omit_redundant_locants=omit_redundant_locants,
        fusion_mode=FusionMode(fusion_mode),
        processes=processes,
        chunksize=chunksize,
        style=resolve_style(
            style,
            enclosing_marks=enclosing_marks,
            thiol_prefix=thiol_prefix,
            locant_placement=locant_placement,
            heterocycle_names=heterocycle_names,
        ),
    )


__all__ = [
    "AtomBinding",
    "BondBinding",
    "DecisionTrace",
    "DescribedComponent",
    "Description",
    "DescriptionTokenSummary",
    "FunctionalGroupMetadata",
    "FusionMode",
    "HumanDescription",
    "NameAnalysis",
    "NamingEngine",
    "NamingIntent",
    "NamingRequest",
    "NamingResult",
    "NamingStyle",
    "NomenclatureOperation",
    "OperationClass",
    "OpsinCheck",
    "OpsinStatus",
    "RULES",
    "STYLE_PRESETS",
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
    "resolve_style",
    "verify_with_opsin",
]
