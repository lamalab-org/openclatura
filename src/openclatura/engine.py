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

from .graph_io import get_connected_components, read_smiles
from .molecule import DecisionTrace, NameAnalysis, TracePhase
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
    """

    smiles: str
    include_trace: bool = False
    verify_opsin: bool = False
    token_debug: bool = False


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

    @property
    def ok(self) -> bool:
        """``True`` when naming produced a non-empty name and did not error."""

        return self.error is None and bool(self.name)

    @property
    def verified(self) -> bool:
        """``True`` when an OPSIN round-trip was requested and matched."""

        return self.opsin_check is not None and self.opsin_check.ok

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
        if include_trace:
            payload["trace_segments"] = self.trace_segments
            payload["substituent_tree"] = self.substituent_tree
        if self.opsin_check is not None:
            payload["opsin_check"] = self.opsin_check.to_dict()
        return payload


class NamingEngine:
    """Facade for structure-to-IUPAC naming.

    The methods currently delegate component naming to the existing
    implementation in ``namer.py``.  Keeping that delegation behind an engine
    boundary lets the next migration stages replace internals without changing
    public callers.
    """

    def name(self, smiles: str) -> str:
        """Return the generated name for ``smiles``."""

        return self.run(NamingRequest(smiles=smiles)).name

    def name_smiles(self, smiles: str) -> str:
        """Compatibility alias for the legacy public API name."""

        return self.name(smiles)

    def name_with_trace(self, smiles: str) -> tuple[str, list[dict]]:
        """Return the generated name and assembly trace segments."""

        result = self.run(NamingRequest(smiles=smiles, include_trace=True))
        return result.name, result.trace_segments

    def name_smiles_with_trace(self, smiles: str) -> tuple[str, list[dict]]:
        """Compatibility alias for the legacy public API name."""

        return self.name_with_trace(smiles)

    def analyze(self, smiles: str, *, token_debug: bool = False) -> NameAnalysis:
        """Return the full explainable naming analysis for ``smiles``."""

        result = self.run(NamingRequest(smiles=smiles, include_trace=True, token_debug=token_debug))
        if result.analysis is None:
            return NameAnalysis(result.name, result.trace_segments, result.decisions, result.substituent_tree)
        return result.analysis

    def analyze_smiles(self, smiles: str, *, token_debug: bool = False) -> NameAnalysis:
        """Compatibility alias for the legacy public API name."""

        return self.analyze(smiles, token_debug=token_debug)

    def run(self, request: NamingRequest) -> NamingResult:
        """Execute a naming request, never raising for naming failures.

        Internal naming errors are captured as ``result.error`` rather than
        propagated, which makes the batch API safe to call on noisy
        datasets.
        """

        try:
            if request.include_trace:
                analysis = self._analyze(request.smiles, token_debug=request.token_debug)
                rules, hints = _extract_rules_hit(analysis.trace_segments)
                result = NamingResult(
                    name=analysis.name,
                    smiles=request.smiles,
                    trace_segments=analysis.trace_segments,
                    substituent_tree=analysis.substituent_tree,
                    decisions=analysis.decisions,
                    analysis=analysis,
                    rules_hit=rules,
                    rule_hints=hints,
                )
            else:
                result = NamingResult(name=self._name(request.smiles), smiles=request.smiles)
        except Exception as exc:  # noqa: BLE001 - intentionally permissive boundary
            return NamingResult(name="", smiles=request.smiles, error=f"{type(exc).__name__}: {exc}")

        if request.verify_opsin:
            check = verify_with_opsin(result.name, request.smiles)
            result = replace(result, opsin_check=check)

        return result

    def name_many(
        self,
        smiles_iter: Iterable[str],
        *,
        include_trace: bool = False,
        verify_opsin: bool = False,
        token_debug: bool = False,
        processes: int | None | str = 1,
        chunksize: int = 64,
    ) -> list[NamingResult]:
        """Name a batch of SMILES, optionally in parallel.

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
                    NamingRequest(
                        smiles=s,
                        include_trace=include_trace,
                        verify_opsin=verify_opsin,
                        token_debug=token_debug,
                    )
                )
                for s in smiles_list
            ]

        worker_count = processes if processes is not None else os.cpu_count() or 1
        return _run_parallel(
            smiles_list,
            include_trace=include_trace,
            verify_opsin=verify_opsin,
            token_debug=token_debug,
            processes=worker_count,
            chunksize=chunksize,
        )

    def _name(self, smiles: str) -> str:
        mol = read_smiles(smiles)
        if not mol.atoms:
            return ""

        names = []
        for component in get_connected_components(mol):
            component_name = self._name_component(mol, component)
            if component_name:
                names.append(component_name)
        names.sort(key=self._component_sort_key)
        return " ".join(names)

    def _analyze(self, smiles: str, *, token_debug: bool = False) -> NameAnalysis:
        decisions = DecisionTrace()
        mol = read_smiles(smiles)
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
            )
            if component_name:
                named_components.append((component_name, trace, tree))

        named_components.sort(key=lambda item: self._component_sort_key(item[0]))
        final_name = " ".join(name for name, _, _ in named_components)
        trace_segments = []
        substituent_tree = []
        for _, trace, tree in named_components:
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
            data={"name": final_name, "components": [name for name, _, _ in named_components]},
        )
        return NameAnalysis(
            name=final_name,
            trace_segments=trace_segments,
            decisions=decisions.steps,
            substituent_tree=substituent_tree,
            operations=infer_operations(decisions.steps, trace_segments),
        )

    @staticmethod
    def _component_sort_key(name: str) -> tuple[int, str]:
        return (0 if name in SALT_METAL_NAMES else 1, name)

    @staticmethod
    def _name_component(*args: Any, **kwargs: Any):
        from .namer import name_component

        return name_component(*args, **kwargs)


DEFAULT_NAMING_ENGINE = NamingEngine()


# --- multiprocessing helpers ---------------------------------------------


def _name_one_for_worker(args: tuple[str, bool, bool, bool]) -> NamingResult:
    smiles, include_trace, verify_opsin, token_debug = args
    return DEFAULT_NAMING_ENGINE.run(
        NamingRequest(smiles=smiles, include_trace=include_trace, verify_opsin=verify_opsin, token_debug=token_debug)
    )


def _run_parallel(
    smiles_list: list[str],
    *,
    include_trace: bool,
    verify_opsin: bool,
    token_debug: bool,
    processes: int,
    chunksize: int,
) -> list[NamingResult]:
    # Imported lazily so the simple `import openclatura` path stays light.
    from concurrent.futures import ProcessPoolExecutor

    payload = [(s, include_trace, verify_opsin, token_debug) for s in smiles_list]
    with ProcessPoolExecutor(max_workers=processes) as ex:
        return list(ex.map(_name_one_for_worker, payload, chunksize=chunksize))
