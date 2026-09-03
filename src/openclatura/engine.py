"""Public naming engine facade.

The legacy module-level API in :mod:`openclatura.namer` is kept for
compatibility.  This module provides the architectural seam for the staged
refactor: callers can use one engine object while the internals are migrated
from legacy functions to typed pipeline stages.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .fusion.context import current_fusion_mode, reset_fusion_mode, set_fusion_mode
from .fusion.model import FusionMode
from .graph_io import get_connected_components, read_rdkit_mol, read_smiles
from .molecule import DecisionTrace, Molecule, NameAnalysis, TracePhase
from .name_assembly import set_token_span_building
from .namer_config import SALT_METAL_NAMES
from .operations import infer_operations
from .opsin_verify import OpsinCheck, verify_with_opsin
from .trace_helpers import attach_main_parent_decisions, trace_decision

# Blue-Book-style rule identifiers (P-12, P-23.2.5, P-66.1.2.4 etc.).
_RULE_ID_PATTERN = re.compile(r"P-\d+(?:\.\d+)*")


def _extract_rules_hit(trace_segments: Sequence[dict]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Pull rule identifiers and human-readable hints out of trace segments.

    Returns a pair ``(rules, hints)`` where ``rules`` are de-duplicated and
    sort-stable rule IDs (``P-XX[.Y[.Z]]``) and ``hints`` are the full
    one-line rule_hint strings the trace emitted.
    """

    rules: list[str] = []
    rule_set: set[str] = set()
    hints: list[str] = []
    hint_set: set[str] = set()
    for seg in trace_segments:
        hint = seg.get("rule_hint") if isinstance(seg, dict) else None
        if not hint:
            continue
        if hint not in hint_set:
            hint_set.add(hint)
            hints.append(hint)
        for match in _RULE_ID_PATTERN.findall(hint):
            if match not in rule_set:
                rule_set.add(match)
                rules.append(match)
    return tuple(rules), tuple(hints)


@dataclass(frozen=True)
class NamingRequest:
    """Input options for a molecule naming run.

    ``include_trace`` toggles emission of the explainable analysis fields
    (``trace_segments``, ``decisions``, ``rules_hit``, ``rule_hints``,
    ``analysis``).  ``verify_opsin`` toggles a round-trip check that feeds
    the generated name back through OPSIN and compares the canonical
    SMILES to the input.  Verification is graceful when py2opsin or Java
    are missing (see :class:`openclatura.opsin_verify.OpsinCheck`).

    ``omit_redundant_locants`` controls an experimental symmetry proof for
    constitutional locants on simple chain and monocyclic parents. It defaults
    to true because a provably unique locant configuration gives the better
    name, while callers can explicitly disable it for compatibility.

    Structures arrive either as ``smiles`` or as an already-parsed
    ``rdkit_mol`` (``rdkit.Chem.rdchem.Mol``); when a molecule is given, a
    SMILES is only generated if something downstream actually needs one, so
    callers holding an RDKit molecule or an SD record pay no round-trip cost.
    """

    smiles: str = ""
    include_trace: bool = False
    verify_opsin: bool = False
    verify_self: bool = False
    token_debug: bool = False
    omit_redundant_locants: bool = True
    fusion_mode: FusionMode = FusionMode.LEGACY
    rdkit_mol: Any | None = None


@dataclass(frozen=True)
class NamingResult:
    """Named molecule plus optional explainability and verification data.

    Ergonomics:

    - ``str(result)`` → the generated name (empty string on failure).
    - ``bool(result)`` → ``result.ok``.
    - ``result.to_dict()`` → JSON-friendly dict, ready for dataset rows.
    """

    name: str
    smiles: str = ""
    error: str | None = None
    trace_segments: list[dict] = field(default_factory=list)
    substituent_tree: list[dict] = field(default_factory=list)
    decisions: list = field(default_factory=list)
    analysis: NameAnalysis | None = None
    rules_hit: tuple[str, ...] = ()
    rule_hints: tuple[str, ...] = ()
    opsin_check: OpsinCheck | None = None
    self_audit: Any | None = None
    parent_nomenclature: str | None = None
    pin_status: str | None = None
    fusion_support_tier: str | None = None
    proof_source: str | None = None

    @property
    def ok(self) -> bool:
        """``True`` when naming produced a non-empty name and did not error."""

        return self.error is None and bool(self.name)

    @property
    def verified(self) -> bool:
        """``True`` when an OPSIN round-trip was requested and matched."""

        return self.opsin_check is not None and self.opsin_check.ok

    @property
    def self_verified(self) -> bool:
        """``True`` when the OPSIN-free reconstruction self-audit confirmed the name."""

        return self.self_audit is not None and self.self_audit.ok

    def __str__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"NamingResult(name={self.name!r}, smiles={self.smiles!r}, ok={self.ok})"

    def to_dict(self, *, include_trace: bool = False) -> dict:
        """JSON-friendly dict view. Pass ``include_trace=True`` to keep the raw trace."""

        payload: dict = {
            "smiles": self.smiles,
            "name": self.name,
            "ok": self.ok,
            "error": self.error,
            "rules_hit": list(self.rules_hit),
            "rule_hints": list(self.rule_hints),
        }
        if self.parent_nomenclature is not None:
            payload.update(
                {
                    "parent_nomenclature": self.parent_nomenclature,
                    "pin_status": self.pin_status,
                    "fusion_support_tier": self.fusion_support_tier,
                    "proof_source": self.proof_source,
                }
            )
        if include_trace:
            payload["trace_segments"] = self.trace_segments
            payload["substituent_tree"] = self.substituent_tree
        if self.opsin_check is not None:
            payload["opsin_check"] = self.opsin_check.to_dict()
        if self.self_audit is not None:
            payload["self_audit"] = self.self_audit.to_dict()
        return payload


class NamingEngine:
    """Facade for structure-to-IUPAC naming.

    The methods currently delegate component naming to the existing
    implementation in ``namer.py``.  Keeping that delegation behind an engine
    boundary lets the next migration stages replace internals without changing
    public callers.
    """

    def name(self, smiles: str, *, fusion_mode: FusionMode | str = FusionMode.LEGACY) -> str:
        """Return the generated name for ``smiles``."""

        return self.run(NamingRequest(smiles=smiles, fusion_mode=FusionMode(fusion_mode))).name

    def name_smiles(self, smiles: str, *, fusion_mode: FusionMode | str = FusionMode.LEGACY) -> str:
        """Compatibility alias for the legacy public API name."""

        return self.name(smiles, fusion_mode=fusion_mode)

    def name_rdkit_mol(
        self, rdkit_mol: Any, *, fusion_mode: FusionMode | str = FusionMode.LEGACY
    ) -> str:
        """Return the generated name for an existing ``rdkit.Chem.rdchem.Mol``."""

        return self.run(NamingRequest(rdkit_mol=rdkit_mol, fusion_mode=FusionMode(fusion_mode))).name

    def name_rdkit_mol_with_trace(
        self, rdkit_mol: Any, *, fusion_mode: FusionMode | str = FusionMode.LEGACY
    ) -> tuple[str, list[dict]]:
        """Return the generated name and assembly trace for an RDKit molecule."""

        result = self.run(
            NamingRequest(rdkit_mol=rdkit_mol, include_trace=True, fusion_mode=FusionMode(fusion_mode))
        )
        return result.name, result.trace_segments

    def analyze_rdkit_mol(
        self,
        rdkit_mol: Any,
        *,
        token_debug: bool = False,
        fusion_mode: FusionMode | str = FusionMode.LEGACY,
    ) -> NameAnalysis:
        """Return the full explainable naming analysis for an RDKit molecule."""

        result = self.run(
            NamingRequest(
                rdkit_mol=rdkit_mol,
                include_trace=True,
                token_debug=token_debug,
                fusion_mode=FusionMode(fusion_mode),
            )
        )
        if result.analysis is None:
            return NameAnalysis(result.name, result.trace_segments, result.decisions, result.substituent_tree)
        return result.analysis

    def name_with_trace(
        self, smiles: str, *, fusion_mode: FusionMode | str = FusionMode.LEGACY
    ) -> tuple[str, list[dict]]:
        """Return the generated name and assembly trace segments."""

        result = self.run(
            NamingRequest(smiles=smiles, include_trace=True, fusion_mode=FusionMode(fusion_mode))
        )
        return result.name, result.trace_segments

    def name_smiles_with_trace(
        self, smiles: str, *, fusion_mode: FusionMode | str = FusionMode.LEGACY
    ) -> tuple[str, list[dict]]:
        """Compatibility alias for the legacy public API name."""

        return self.name_with_trace(smiles, fusion_mode=fusion_mode)

    def analyze(
        self,
        smiles: str,
        *,
        token_debug: bool = False,
        fusion_mode: FusionMode | str = FusionMode.LEGACY,
    ) -> NameAnalysis:
        """Return the full explainable naming analysis for ``smiles``."""

        result = self.run(
            NamingRequest(
                smiles=smiles,
                include_trace=True,
                token_debug=token_debug,
                fusion_mode=FusionMode(fusion_mode),
            )
        )
        if result.analysis is None:
            return NameAnalysis(result.name, result.trace_segments, result.decisions, result.substituent_tree)
        return result.analysis

    def analyze_smiles(
        self,
        smiles: str,
        *,
        token_debug: bool = False,
        fusion_mode: FusionMode | str = FusionMode.LEGACY,
    ) -> NameAnalysis:
        """Compatibility alias for the legacy public API name."""

        return self.analyze(smiles, token_debug=token_debug, fusion_mode=fusion_mode)

    def run(self, request: NamingRequest) -> NamingResult:
        """Execute a naming request, never raising for naming failures.

        Internal naming errors are captured as ``result.error`` rather than
        propagated, which makes the batch API safe to call on noisy
        datasets.
        """

        # Token spans feed only the trace/analysis output; skip building them on the
        # pure-name path so the common API does not pay for diagnostics it discards.
        need_analysis = request.include_trace or request.verify_opsin
        previous_span_building = set_token_span_building(need_analysis)
        fusion_mode_token = (
            None if request.fusion_mode is current_fusion_mode() else set_fusion_mode(request.fusion_mode)
        )

        # The OPSIN-free self-audit rebuilds each component from its name while it
        # is being generated, so its capture hook must wrap the naming call.
        if request.verify_self:
            from .audit import capture_component_audits

            audit_cm = capture_component_audits()
        else:
            from contextlib import nullcontext

            audit_cm = nullcontext([])

        try:
            with audit_cm as component_audits:
                mol, smiles = self._prepare_input(request)
                if need_analysis:
                    analysis = self._analyze(
                        mol,
                        smiles=smiles,
                        token_debug=request.token_debug,
                        omit_redundant_locants=request.omit_redundant_locants,
                    )
                    rules, hints = _extract_rules_hit(analysis.trace_segments)
                    result = NamingResult(
                        name=analysis.name,
                        smiles=smiles,
                        trace_segments=analysis.trace_segments,
                        substituent_tree=analysis.substituent_tree,
                        decisions=analysis.decisions,
                        analysis=analysis,
                        rules_hit=rules,
                        rule_hints=hints,
                        **_fusion_result_metadata(analysis.decisions),
                    )
                else:
                    result = NamingResult(
                        name=self._name(mol, omit_redundant_locants=request.omit_redundant_locants),
                        smiles=smiles,
                    )
        except Exception as exc:  # noqa: BLE001 - intentionally permissive boundary
            return NamingResult(name="", smiles=request.smiles, error=f"{type(exc).__name__}: {exc}")
        finally:
            if fusion_mode_token is not None:
                reset_fusion_mode(fusion_mode_token)
            set_token_span_building(previous_span_building)

        if request.verify_opsin:
            check = verify_with_opsin(result.name, result.smiles)
            result = replace(result, opsin_check=check)

        if request.verify_self:
            from .audit import aggregate_audits

            result = replace(result, self_audit=aggregate_audits(list(component_audits)))

        return result

    def name_many(
        self,
        smiles_iter: Iterable[str | Any],
        *,
        include_trace: bool = False,
        verify_opsin: bool = False,
        verify_self: bool = False,
        token_debug: bool = False,
        omit_redundant_locants: bool = True,
        fusion_mode: FusionMode | str = FusionMode.LEGACY,
        processes: int | None | str = 1,
        chunksize: int = 64,
    ) -> list[NamingResult]:
        """Name a batch of SMILES (or RDKit molecules), optionally in parallel.

        ``processes=1`` runs in the current process (good default for
        notebooks and short scripts). ``processes=None`` uses all CPU
        cores. ``processes>1`` uses that many worker processes. Errors
        during naming are captured per-row as ``result.error`` rather
        than propagated, so a single bad SMILES cannot stop the batch.
        """

        if processes == "auto":
            processes = None
        smiles_list = list(smiles_iter)
        if processes == 1:
            return [
                self.run(
                    _request_for(
                        item,
                        include_trace=include_trace,
                        verify_opsin=verify_opsin,
                        verify_self=verify_self,
                        token_debug=token_debug,
                        omit_redundant_locants=omit_redundant_locants,
                        fusion_mode=fusion_mode,
                    )
                )
                for item in smiles_list
            ]

        worker_count = processes if processes is not None else os.cpu_count() or 1
        return _run_parallel(
            smiles_list,
            include_trace=include_trace,
            verify_opsin=verify_opsin,
            verify_self=verify_self,
            token_debug=token_debug,
            omit_redundant_locants=omit_redundant_locants,
            fusion_mode=fusion_mode,
            processes=worker_count,
            chunksize=chunksize,
        )

    @staticmethod
    def _prepare_input(request: NamingRequest) -> tuple[Molecule, str]:
        """Build the internal graph from whichever input the request carries.

        For RDKit input the SMILES stays empty unless it is needed (OPSIN
        verification), so the common case never pays for ``MolToSmiles``.
        """

        if request.rdkit_mol is None:
            return read_smiles(request.smiles), request.smiles

        smiles = request.smiles
        if not smiles and request.verify_opsin:
            from rdkit import Chem

            smiles = Chem.MolToSmiles(request.rdkit_mol)
        return read_rdkit_mol(request.rdkit_mol), smiles

    def _name(self, mol: Molecule, *, omit_redundant_locants: bool = True) -> str:
        if not mol.atoms:
            return ""

        names = []
        for component in get_connected_components(mol):
            component_name = self._name_component(
                mol,
                component,
                omit_redundant_locants=omit_redundant_locants,
            )
            if component_name:
                names.append((component_name, _component_charge(mol, component)))
        names.sort(key=lambda item: self._component_sort_key(*item))
        return " ".join(_multiply_identical_ions(names))

    def _analyze(
        self,
        mol: Molecule,
        *,
        smiles: str = "",
        token_debug: bool = False,
        omit_redundant_locants: bool = True,
    ) -> NameAnalysis:
        decisions = DecisionTrace()
        trace_decision(
            decisions,
            TracePhase.PARSE,
            "parsed SMILES",
            "RDKit parsing populated the internal Molecule graph used by the namer.",
            atoms=set(mol.atoms.keys()),
            bonds=set(mol.bonds.keys()),
            data={"smiles": smiles, "atom_count": len(mol.atoms), "bond_count": len(mol.bonds)},
        )
        if not mol.atoms:
            return NameAnalysis(name="", trace_segments=[], decisions=decisions.steps)

        components = get_connected_components(mol)
        trace_decision(
            decisions,
            TracePhase.COMPONENT,
            "split molecule into components",
            "Each connected graph component is named independently before final component ordering.",
            atoms=set(mol.atoms.keys()),
            data={"components": [sorted(component) for component in components]},
        )

        named_components = []
        for component in components:
            component_name, trace, tree = self._name_component(
                mol,
                component,
                return_trace=True,
                return_tree=True,
                decision_trace=decisions,
                token_debug=token_debug,
                omit_redundant_locants=omit_redundant_locants,
            )
            if component_name:
                named_components.append((component_name, trace, tree, _component_charge(mol, component)))

        named_components.sort(key=lambda item: self._component_sort_key(item[0], item[3]))
        final_name = " ".join(_multiply_identical_ions([(name, charge) for name, _, _, charge in named_components]))
        trace_segments = []
        substituent_tree = []
        for _, trace, tree, _ in named_components:
            trace_segments.extend(trace)
            if tree:
                substituent_tree.append(tree)
        trace_segments = attach_main_parent_decisions(trace_segments, decisions)
        trace_decision(
            decisions,
            TracePhase.ASSEMBLY,
            "assembled final molecule name",
            "Named components are sorted with supported salt metals first, then joined.",
            atoms=set(mol.atoms.keys()),
            data={"name": final_name, "components": [name for name, _, _, _ in named_components]},
        )
        return NameAnalysis(
            name=final_name,
            trace_segments=trace_segments,
            decisions=decisions.steps,
            substituent_tree=substituent_tree,
            operations=infer_operations(decisions.steps, trace_segments),
        )

    @staticmethod
    def _component_sort_key(name: str, charge: int = 0) -> tuple[int, int, str]:
        """P-72.3: cations are cited before anions; metals first among cations, then alphabetical."""

        charge_rank = 0 if charge > 0 else (1 if charge == 0 else 2)
        return (charge_rank, 0 if name in SALT_METAL_NAMES else 1, name)

    @staticmethod
    def _name_component(*args: Any, **kwargs: Any):
        from .namer import name_component

        return name_component(*args, **kwargs)


DEFAULT_NAMING_ENGINE = NamingEngine()


def _fusion_result_metadata(decisions) -> dict[str, str | None]:
    """Project a traced systematic-fusion decision onto the public result."""

    step = next(
        (
            item
            for item in decisions
            if item.phase == TracePhase.PARENT_SELECTION
            and item.decision == "selected audited systematic fusion parent"
        ),
        None,
    )
    if step is None:
        return {}
    return {
        "parent_nomenclature": step.data.get("parent_nomenclature"),
        "pin_status": step.data.get("pin_status"),
        "fusion_support_tier": step.data.get("fusion_support_tier"),
        "proof_source": step.data.get("proof_source"),
    }


# --- multiprocessing helpers ---------------------------------------------


def _request_for(
    item: str | Any,
    *,
    include_trace: bool,
    verify_opsin: bool,
    verify_self: bool = False,
    token_debug: bool,
    omit_redundant_locants: bool = True,
    fusion_mode: FusionMode | str = FusionMode.LEGACY,
) -> NamingRequest:
    """Build a request from a batch item, which may be a SMILES or an RDKit molecule."""

    kwargs: dict[str, Any] = {"smiles": item} if isinstance(item, str) else {"rdkit_mol": item}
    return NamingRequest(
        include_trace=include_trace,
        verify_opsin=verify_opsin,
        verify_self=verify_self,
        token_debug=token_debug,
        omit_redundant_locants=omit_redundant_locants,
        fusion_mode=FusionMode(fusion_mode),
        **kwargs,
    )


def _name_one_for_worker(args: tuple[str | Any, bool, bool, bool, bool, bool, str]) -> NamingResult:
    item, include_trace, verify_opsin, verify_self, token_debug, omit_redundant_locants, fusion_mode = args
    return DEFAULT_NAMING_ENGINE.run(
        _request_for(
            item,
            include_trace=include_trace,
            verify_opsin=verify_opsin,
            verify_self=verify_self,
            token_debug=token_debug,
            omit_redundant_locants=omit_redundant_locants,
            fusion_mode=fusion_mode,
        )
    )


def _run_parallel(
    smiles_list: list[str | Any],
    *,
    include_trace: bool,
    verify_opsin: bool,
    verify_self: bool = False,
    token_debug: bool,
    omit_redundant_locants: bool = True,
    fusion_mode: FusionMode | str = FusionMode.LEGACY,
    processes: int,
    chunksize: int,
) -> list[NamingResult]:
    # Imported lazily so the simple `import openclatura` path stays light.
    from concurrent.futures import ProcessPoolExecutor

    mode = FusionMode(fusion_mode).value
    payload = [
        (s, include_trace, verify_opsin, verify_self, token_debug, omit_redundant_locants, mode)
        for s in smiles_list
    ]
    with ProcessPoolExecutor(max_workers=processes) as ex:
        return list(ex.map(_name_one_for_worker, payload, chunksize=chunksize))


def _component_charge(mol: Molecule, component: set[int]) -> int:
    return sum(mol.atoms[idx].charge for idx in component)


def _multiply_identical_ions(names: list[tuple[str, int]]) -> list[str]:
    """P-72.3.1: identical ionic components are cited once with a multiplying prefix (diammonium sulfate)."""

    from .rules import multipliers

    out: list[str] = []
    index = 0
    while index < len(names):
        name, charge = names[index]
        count = 1
        while index + count < len(names) and names[index + count] == (name, charge):
            count += 1
        if count > 1 and charge:
            simple = (
                " " not in name
                and "-" not in name
                and (name in SALT_METAL_NAMES or not name.endswith("ium") or name == "ammonium")
            )
            if simple:
                out.append(f"{multipliers.basic(count)}{name}")
            else:
                out.append(f"{multipliers.complex_(count)}({name})")
        else:
            out.extend([name] * count)
        index += count
    return out
