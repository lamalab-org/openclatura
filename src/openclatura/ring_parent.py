"""Structured ring-parent handoff objects.

`RingSystem` is the legacy discovery object.  `RingParent` is the migration
target: one object that keeps descriptor text, numbering candidates, locant
maps, and audit metadata together instead of passing a descriptor string and
plain paths independently.
"""

from dataclasses import dataclass

from .polycycle_topology import RingNumbering
from .ring_renderer import is_von_baeyer_descriptor


def _is_von_baeyer_descriptor(descriptor: str | None) -> bool:
    return is_von_baeyer_descriptor(descriptor)


@dataclass(frozen=True)
class RingParent:
    kind: str
    atoms: frozenset[int]
    descriptor: str | None = None
    descriptor_numbers: tuple[int, ...] = ()
    candidate_paths: tuple[tuple[int, ...], ...] = ()
    numbering_candidates: tuple[RingNumbering, ...] = ()
    selected_numbering: RingNumbering | None = None
    retained_locant_maps: tuple[dict[int, str], ...] = ()
    proof_source: str = "topology"

    @property
    def paths(self) -> list[list[int]]:
        if self.numbering_candidates:
            return [list(numbering.path) for numbering in self.numbering_candidates]
        if self.retained_locant_maps:
            return [
                sorted(locant_map, key=lambda atom: _retained_locant_sort_key(locant_map[atom]))
                for locant_map in self.retained_locant_maps
            ]
        return [list(path) for path in self.candidate_paths]

    @property
    def locant_map(self) -> dict[int, str] | None:
        if self.selected_numbering is None:
            return None
        return self.selected_numbering.locant_map

    @property
    def audit_ok(self) -> bool:
        if self.retained_locant_maps:
            return all(
                set(locant_map) == set(self.atoms) and len(set(locant_map.values())) == len(self.atoms)
                for locant_map in self.retained_locant_maps
            )
        if not self.numbering_candidates:
            return not _is_von_baeyer_descriptor(self.descriptor)
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
            proof_source="descriptor",
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
        if _is_von_baeyer_descriptor(descriptor):
            raise ValueError("von Baeyer RingParent requires audited numbering candidates")
        return cls(
            kind=kind,
            atoms=frozenset(atoms),
            descriptor=descriptor,
            descriptor_numbers=descriptor_numbers,
            candidate_paths=tuple(tuple(path) for path in paths),
        )

    @classmethod
    def from_retained_locant_maps(
        cls,
        *,
        atoms: set[int] | frozenset[int],
        locant_maps: list[dict[int, str]],
    ) -> "RingParent":
        """Build an audited parent proof from exact template isomorphisms."""

        parent = cls(
            kind="retained_polycycle",
            atoms=frozenset(atoms),
            retained_locant_maps=tuple(dict(locant_map) for locant_map in locant_maps),
            proof_source="retained_template",
        )
        if not parent.audit_ok:
            raise ValueError("Retained parent locant maps must be complete bijections.")
        return parent


def _retained_locant_sort_key(locant: str) -> tuple[int, str]:
    digits = ""
    suffix = ""
    for char in str(locant):
        if char.isdigit() and not suffix:
            digits += char
        else:
            suffix += char
    return (int(digits) if digits else 10_000, suffix)
