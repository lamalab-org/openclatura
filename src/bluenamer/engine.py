"""Public naming engine facade.

The legacy module-level API in :mod:`bluenamer.namer` is kept for
compatibility.  This module provides the architectural seam for the staged
refactor: callers can use one engine object while the internals are migrated
from legacy functions to typed pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph_io import get_connected_components, read_smiles
from .molecule import DecisionTrace, NameAnalysis, TracePhase
from .namer_config import SALT_METAL_NAMES
from .operations import infer_operations
from .trace_helpers import trace_decision


@dataclass(frozen=True)
class NamingRequest:
    """Input options for a molecule naming run."""

    smiles: str
    include_trace: bool = False


@dataclass(frozen=True)
class NamingResult:
    """Named molecule plus trace metadata emitted by the engine."""

    name: str
    trace_segments: list[dict] = field(default_factory=list)
    decisions: list = field(default_factory=list)
    analysis: NameAnalysis | None = None


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

    def analyze(self, smiles: str) -> NameAnalysis:
        """Return the full explainable naming analysis for ``smiles``."""

        result = self.run(NamingRequest(smiles=smiles, include_trace=True))
        if result.analysis is None:
            return NameAnalysis(result.name, result.trace_segments, result.decisions)
        return result.analysis

    def analyze_smiles(self, smiles: str) -> NameAnalysis:
        """Compatibility alias for the legacy public API name."""

        return self.analyze(smiles)

    def run(self, request: NamingRequest) -> NamingResult:
        """Execute a naming request."""

        if request.include_trace:
            analysis = self._analyze(request.smiles)
            return NamingResult(
                name=analysis.name,
                trace_segments=analysis.trace_segments,
                decisions=analysis.decisions,
                analysis=analysis,
            )
        return NamingResult(name=self._name(request.smiles))

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

    def _analyze(self, smiles: str) -> NameAnalysis:
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
            component_name, trace = self._name_component(
                mol,
                component,
                return_trace=True,
                decision_trace=decisions,
            )
            if component_name:
                named_components.append((component_name, trace))

        named_components.sort(key=lambda item: self._component_sort_key(item[0]))
        final_name = " ".join(name for name, _ in named_components)
        trace_segments = []
        for _, trace in named_components:
            trace_segments.extend(trace)
        trace_decision(
            decisions,
            TracePhase.ASSEMBLY,
            "assembled final molecule name",
            "Named components are sorted with supported salt metals first, then joined.",
            atoms=set(mol.atoms.keys()),
            data={"name": final_name, "components": [name for name, _ in named_components]},
        )
        return NameAnalysis(
            name=final_name,
            trace_segments=trace_segments,
            decisions=decisions.steps,
            operations=infer_operations(decisions.steps, trace_segments),
        )

    @staticmethod
    def _component_sort_key(name: str) -> tuple[int, str]:
        return (0 if name in SALT_METAL_NAMES else 1, name)

    @staticmethod
    def _name_component(*args, **kwargs):
        from .namer import name_component

        return name_component(*args, **kwargs)


DEFAULT_NAMING_ENGINE = NamingEngine()
