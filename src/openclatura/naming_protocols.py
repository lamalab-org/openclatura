"""Shared callable contracts for recursive structure naming."""

from typing import Literal, Protocol, overload

from .assembly_parts import RenderedSubstituentText
from .molecule import DecisionTrace, Molecule


class RecursiveSubgraphNamer(Protocol):
    """Name a branch, optionally returning trace segments and its tree node."""

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[False] = False,
        return_tree: Literal[False] = False,
        decision_trace: DecisionTrace | None = None,
    ) -> RenderedSubstituentText: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[True],
        return_tree: Literal[False] = False,
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[RenderedSubstituentText, list[dict]]: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[True],
        return_tree: Literal[True],
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[RenderedSubstituentText, list[dict], dict | None]: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[False] = False,
        return_tree: Literal[True],
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[RenderedSubstituentText, dict | None]: ...
