"""Structured ring-parent handoff objects.

`RingSystem` is the legacy discovery object.  `RingParent` is the migration
target: one object that keeps descriptor text, numbering candidates, locant
maps, and audit metadata together instead of passing a descriptor string and
plain paths independently.
"""

from dataclasses import dataclass

from .polycycle_topology import RingNumbering
from .ring_renderer import is_von_baeyer_descriptor


@dataclass(frozen=True)
class RingParent:
    kind: str
    atoms: frozenset[int]
    descriptor: str | None = None
    descriptor_numbers: tuple[int, ...] = ()
    candidate_paths: tuple[tuple[int, ...], ...] = ()
    numbering_candidates: tuple[RingNumbering, ...] = ()
    selected_numbering: RingNumbering | None = None

    @property
    def paths(self) -> list[list[int]]:
        if self.numbering_candidates:
            return [list(numbering.path) for numbering in self.numbering_candidates]
        return [list(path) for path in self.candidate_paths]

    @property
    def locant_map(self) -> dict[int, str] | None:
        if self.selected_numbering is None:
            return None
        return self.selected_numbering.locant_map

    @property
    def audit_ok(self) -> bool:
        if not self.numbering_candidates:
            return not is_von_baeyer_descriptor(self.descriptor)
        return all(numbering.audit_ok for numbering in self.numbering_candidates)

    @classmethod
    def from_numberings(
        cls,
        *,
        kind: str,
        atoms: set[int] | frozenset[int],
        descriptor: str | None,
        descriptor_numbers: tuple[int, ...],
        numberings: list[RingNumbering],
        selected_path: list[int] | tuple[int, ...] | None = None,
    ) -> "RingParent":
        selected = None
        if selected_path is not None:
            selected_tuple = tuple(selected_path)
            selected = next((numbering for numbering in numberings if numbering.path == selected_tuple), None)
        if selected is None and numberings:
            selected = numberings[0]
        return cls(
            kind=kind,
            atoms=frozenset(atoms),
            descriptor=descriptor,
            descriptor_numbers=descriptor_numbers,
            candidate_paths=tuple(numbering.path for numbering in numberings),
            numbering_candidates=tuple(numberings),
            selected_numbering=selected,
        )

    @classmethod
    def from_paths(
        cls,
        *,
        kind: str,
        atoms: set[int] | frozenset[int],
        descriptor: str | None,
        paths: list[list[int]] | tuple[tuple[int, ...], ...],
        descriptor_numbers: tuple[int, ...] = (),
    ) -> "RingParent":
        if is_von_baeyer_descriptor(descriptor):
            raise ValueError("von Baeyer RingParent requires audited numbering candidates")
        return cls(
            kind=kind,
            atoms=frozenset(atoms),
            descriptor=descriptor,
            descriptor_numbers=descriptor_numbers,
            candidate_paths=tuple(tuple(path) for path in paths),
        )
