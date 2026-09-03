"""Graph-backed registry of components eligible for fusion nomenclature.

The JSON table in :mod:`openclatura.data` contains nomenclature policy only.
Connectivity and local numbering remain owned by the retained graph-template
registry, so this module never identifies a component from a textual name,
SMILES/SMARTS pattern, or drawing coordinates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from typing import Any

from ..molecule import Molecule
from ..naming_data import load_json_table
from ..polycycle_topology import normalize_edge
from ..retained_fused_templates import (
    RetainedGraphTemplate,
    match_retained_graph_template_maps,
    molecule_graph_topology_key,
    retained_graph_template_topology_key,
    retained_graph_templates,
    validate_retained_fused_template,
)
from ..retained_graph_model import monocyclic_graph_template
from .model import FusionComponentMatch, FusionComponentSpec

SUPPORTED_SCHEMA_VERSION = 1


class FusionComponentRole(StrEnum):
    """A nomenclatural role for which a fusion component may be eligible."""

    PARENT = "parent"
    ATTACHED = "attached"


@dataclass(frozen=True, slots=True)
class RegisteredFusionComponent:
    """One validated policy row joined to its graph templates."""

    spec: FusionComponentSpec
    template_names: tuple[str, ...]
    templates: tuple[RetainedGraphTemplate, ...]
    omit_attached_locants: bool = False

    def eligible_for(self, role: FusionComponentRole | str) -> bool:
        role = FusionComponentRole(role)
        return self.spec.usable_as_parent if role is FusionComponentRole.PARENT else self.spec.usable_as_attached

    def spec_for_template(self, template_name: str) -> FusionComponentSpec:
        """Bind this component policy to the exact matched graph variant."""

        if not template_name:
            return self.spec
        template = next((item for item in self.templates if item.name == template_name), None)
        if template is None:
            raise KeyError(f"component {self.spec.key!r} has no template {template_name!r}")
        return replace(self.spec, template=template)


@dataclass(frozen=True, slots=True)
class _Face:
    id: int
    atoms: frozenset[int]
    edges: frozenset[tuple[int, int]]


class FusionComponentRegistry:
    """Validated, topology-indexed fusion-component registry."""

    def __init__(
        self,
        version: str,
        *,
        templates: Iterable[RetainedGraphTemplate] | None = None,
    ) -> None:
        if not isinstance(version, str) or not version.strip():
            raise ValueError("fusion component registry version must not be empty")
        self.version = version
        source_templates = retained_graph_templates(include_disabled=True) if templates is None else tuple(templates)
        self._template_by_name = {template.name: template for template in source_templates}
        if len(self._template_by_name) != len(source_templates):
            raise ValueError("retained graph template names must be unique")
        self._components: list[RegisteredFusionComponent] = []
        self._by_key: dict[str, RegisteredFusionComponent] = {}
        self._claimed_template_names: set[str] = set()
        self._topology_index: dict[tuple[int, tuple], list[RegisteredFusionComponent]] = {}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> FusionComponentRegistry:
        """Build and validate a registry from the serialized policy table."""

        schema_version = data.get("schema_version")
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported fusion component schema version {schema_version!r}; expected {SUPPORTED_SCHEMA_VERSION}"
            )
        rows = data.get("components")
        if not isinstance(rows, list):
            raise ValueError("fusion component data must contain a components list")
        generated = _generated_component_templates(data.get("generated_components", ()))
        registry = cls(
            _required_text(data, "registry_version"),
            templates=(*retained_graph_templates(include_disabled=True), *generated),
        )
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("every fusion component must be a mapping")
            registry.register(row)
        return registry

    @property
    def components(self) -> tuple[RegisteredFusionComponent, ...]:
        return tuple(self._components)

    @property
    def specs(self) -> tuple[FusionComponentSpec, ...]:
        return tuple(component.spec for component in self._components)

    @property
    def by_key(self) -> Mapping[str, RegisteredFusionComponent]:
        return dict(self._by_key)

    def register(self, row: Mapping[str, Any]) -> FusionComponentSpec:
        """Validate and register one data-backed component policy row."""

        key = _required_text(row, "key")
        if key in self._by_key:
            raise ValueError(f"duplicate fusion component key {key!r}")

        template_names = _template_names(row, default=key)
        duplicates = self._claimed_template_names.intersection(template_names)
        if duplicates:
            raise ValueError(f"duplicate fusion component template name {min(duplicates)!r}")
        missing = tuple(name for name in template_names if name not in self._template_by_name)
        if missing:
            raise ValueError(f"fusion component {key!r} references unknown templates: {', '.join(missing)}")
        templates = tuple(self._template_by_name[name] for name in template_names)
        for template in templates:
            validate_retained_fused_template(template)
        if any(template.locants != templates[0].locants for template in templates[1:]):
            raise ValueError(f"fusion component {key!r} template variants use different local locants")

        allow_parent = _required_bool(row, "allow_as_parent")
        allow_attached = _required_bool(row, "allow_as_attached")
        if not allow_parent and not allow_attached:
            raise ValueError(f"fusion component {key!r} is not eligible for any role")
        primary = templates[0]
        attached_prefix = _optional_text(row, "attached_prefix") or primary.attached_prefix
        if allow_attached and not attached_prefix:
            raise ValueError(f"attached fusion component {key!r} requires an attached_prefix")
        rule = _required_text(row, "rule")
        parent_name = _optional_text(row, "parent_name") or primary.output_name
        omit_attached_locants = row.get("omit_attached_locants", False)
        if type(omit_attached_locants) is not bool:
            raise ValueError("omit_attached_locants must be a boolean")

        spec = _component_spec(
            key=key,
            parent_name=parent_name,
            attached_prefix=attached_prefix or "",
            template=primary,
            usable_as_parent=allow_parent,
            usable_as_attached=allow_attached,
            rule_reference=rule,
            seniority_override=_optional_nonnegative_int(row, "seniority_override"),
            horizontal_ring_count=_horizontal_ring_count(row, primary),
        )
        component = RegisteredFusionComponent(spec, template_names, templates, omit_attached_locants)
        self._components.append(component)
        self._by_key[key] = component
        self._claimed_template_names.update(template_names)
        for template in templates:
            index_key = (len(template.rings), retained_graph_template_topology_key(template))
            self._topology_index.setdefault(index_key, []).append(component)
        return spec

    def match_faces(
        self,
        mol: Molecule,
        faces: object,
        *,
        role: FusionComponentRole | str | None = None,
        input_to_skeleton_atom: Mapping[int, int] | None = None,
    ) -> tuple[FusionComponentMatch, ...]:
        """Return every exact local numbering for components covering faces.

        Face subsets are connected through shared molecular edges. Cheap graph
        invariants select candidate templates; the retained graph matcher then
        proves connectivity, elements, charges, and every local locant map.
        """

        requested_role = FusionComponentRole(role) if role is not None else None
        normalized_faces = _normalize_faces(faces)
        if not normalized_faces:
            return ()
        unknown = set().union(*(face.atoms for face in normalized_faces)) - mol.atoms.keys()
        if unknown:
            raise KeyError(f"unknown face atom ids: {sorted(unknown)}")
        if missing_edges := {edge for face in normalized_faces for edge in face.edges if mol.get_bond(*edge) is None}:
            raise ValueError(f"face cycles contain non-bonded atom pairs: {sorted(missing_edges)}")
        skeleton_map = dict(input_to_skeleton_atom or ((atom, atom) for atom in mol.atoms))
        missing_skeleton_atoms = set().union(*(face.atoms for face in normalized_faces)) - skeleton_map.keys()
        if missing_skeleton_atoms:
            raise ValueError(f"missing skeleton atom mappings: {sorted(missing_skeleton_atoms)}")

        ring_counts = frozenset(count for count, _ in self._topology_index)
        matches: list[FusionComponentMatch] = []
        seen: set[tuple[str, str, frozenset[int], tuple[tuple[str, int], ...]]] = set()
        occurrence_id = 0
        for subset in _connected_face_subsets(normalized_faces, ring_counts):
            atom_ids = set().union(*(face.atoms for face in subset))
            topology_key = molecule_graph_topology_key(mol, atom_ids)
            candidates = self._topology_index.get((len(subset), topology_key), ())
            for component in candidates:
                if requested_role is not None and not component.eligible_for(requested_role):
                    continue
                for template in component.templates:
                    if retained_graph_template_topology_key(template) != topology_key:
                        continue
                    for exact in match_retained_graph_template_maps(
                        mol,
                        atom_ids,
                        template,
                        allow_nonaromatic=True,
                    ):
                        if not _template_rings_match_faces(template, exact.locant_to_atom, subset):
                            continue
                        local_to_input = tuple((locant, exact.locant_to_atom[locant]) for locant in template.locants)
                        identity = (
                            component.spec.key,
                            template.name,
                            frozenset(face.id for face in subset),
                            local_to_input,
                        )
                        if identity in seen:
                            continue
                        seen.add(identity)
                        local_to_skeleton = tuple((locant, skeleton_map[atom]) for locant, atom in local_to_input)
                        matches.append(
                            FusionComponentMatch(
                                occurrence_id=occurrence_id,
                                spec_key=component.spec.key,
                                covered_face_ids=identity[2],
                                local_to_input_atom=local_to_input,
                                local_to_skeleton_atom=local_to_skeleton,
                                topology_key=topology_key,
                                template_name=template.name,
                            )
                        )
                        occurrence_id += 1
        return tuple(matches)

    def spec_for_match(self, match: FusionComponentMatch) -> FusionComponentSpec:
        """Resolve policy and exact graph data for a proven occurrence."""

        component = self._by_key.get(match.spec_key)
        if component is None:
            raise KeyError(f"unknown fusion component {match.spec_key!r}")
        return component.spec_for_template(match.template_name)


def _component_spec(
    *,
    key: str,
    parent_name: str,
    attached_prefix: str,
    template: RetainedGraphTemplate,
    usable_as_parent: bool,
    usable_as_attached: bool,
    rule_reference: str,
    seniority_override: int | None,
    horizontal_ring_count: int,
) -> FusionComponentSpec:
    return FusionComponentSpec(
        key=key,
        parent_name=parent_name,
        attached_prefix=attached_prefix,
        template=template,
        usable_as_parent=usable_as_parent,
        usable_as_attached=usable_as_attached,
        rule_reference=rule_reference,
        seniority_override=seniority_override,
        horizontal_ring_count=horizontal_ring_count,
    )


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_text(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string when supplied")
    return value


def _required_bool(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _optional_nonnegative_int(row: Mapping[str, Any], field: str) -> int | None:
    value = row.get(field)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer when supplied")
    return value


def _horizontal_ring_count(row: Mapping[str, Any], template: RetainedGraphTemplate) -> int:
    value = _optional_nonnegative_int(row, "horizontal_ring_count")
    if value is not None:
        return value
    return 1 if len(template.rings) == 1 else 0


def _template_names(row: Mapping[str, Any], *, default: str) -> tuple[str, ...]:
    values = row.get("template_names", [default])
    if not isinstance(values, list) or not values:
        raise ValueError("template_names must be a non-empty list")
    names = tuple(values)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("template_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("template_names must be unique")
    return names


def _generated_component_templates(rows: object) -> tuple[RetainedGraphTemplate, ...]:
    """Construct graph templates declared by supported data generators."""

    if not isinstance(rows, list):
        raise ValueError("generated_components must be a list")
    templates = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every generated component must be a mapping")
        generator = _required_text(row, "generator")
        if generator != "monocyclic_carbocycle":
            raise ValueError(f"unsupported fusion component generator {generator!r}")
        ring_size = row.get("ring_size")
        if type(ring_size) is not int or ring_size < 3:
            raise ValueError("generated monocyclic carbocycle ring_size must be at least three")
        templates.append(
            monocyclic_graph_template(
                name=_required_text(row, "key"),
                ring_size=ring_size,
                bond_class=_required_text(row, "bond_class"),
                pin=_optional_bool(row, "pin_component", default=True),
            )
        )
    return tuple(templates)


def _optional_bool(row: Mapping[str, Any], field: str, *, default: bool) -> bool:
    value = row.get(field, default)
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _normalize_faces(faces: object) -> tuple[_Face, ...]:
    values = getattr(faces, "faces", faces)
    if not isinstance(values, Iterable):
        raise TypeError("faces must be an iterable or expose a faces iterable")
    normalized: list[_Face] = []
    for position, face in enumerate(values):
        atom_cycle = getattr(face, "atom_cycle", getattr(face, "atoms", face))
        if not isinstance(atom_cycle, Sequence):
            atom_cycle = tuple(atom_cycle)
        atoms = tuple(int(atom) for atom in atom_cycle)
        if len(atoms) < 3 or len(atoms) != len(set(atoms)):
            raise ValueError("every fusion face must be a simple cycle")
        face_id = int(getattr(face, "id", position))
        edges = frozenset(normalize_edge(left, right) for left, right in zip(atoms, atoms[1:] + atoms[:1]))
        normalized.append(_Face(face_id, frozenset(atoms), edges))
    if len({face.id for face in normalized}) != len(normalized):
        raise ValueError("fusion face ids must be unique")
    return tuple(sorted(normalized, key=lambda face: face.id))


def _connected_face_subsets(faces: tuple[_Face, ...], sizes: frozenset[int]) -> tuple[tuple[_Face, ...], ...]:
    if not sizes:
        return ()
    maximum = min(max(sizes), len(faces))
    adjacent = {
        index: frozenset(
            other for other in range(len(faces)) if other != index and faces[index].edges & faces[other].edges
        )
        for index in range(len(faces))
    }
    frontier = {frozenset((index,)) for index in range(len(faces))}
    subsets: list[tuple[_Face, ...]] = []
    for size in range(1, maximum + 1):
        if size in sizes:
            subsets.extend(tuple(faces[index] for index in sorted(indices)) for indices in sorted(frontier, key=tuple))
        next_frontier: set[frozenset[int]] = set()
        for indices in frontier:
            candidates = set().union(*(adjacent[index] for index in indices)) - indices
            for candidate in candidates:
                next_frontier.add(indices | {candidate})
        frontier = {indices for indices in next_frontier if len(indices) == size + 1}
        if not frontier:
            break
    return tuple(subsets)


def _template_rings_match_faces(
    template: RetainedGraphTemplate,
    locant_to_atom: Mapping[str, int],
    faces: tuple[_Face, ...],
) -> bool:
    mapped_rings = {frozenset(locant_to_atom[locant] for locant in ring) for ring in template.rings}
    return mapped_rings == {face.atoms for face in faces}


@cache
def fusion_component_registry() -> FusionComponentRegistry:
    """Return the process-wide registry loaded from checked-in data."""

    return FusionComponentRegistry.from_data(load_json_table("fusion_components.json"))


def version() -> str:
    """Return the checked-in fusion-component registry version."""

    return fusion_component_registry().version


def register(row: Mapping[str, Any]) -> FusionComponentSpec:
    """Register one component in the process-wide registry."""

    return fusion_component_registry().register(row)


def match_faces(
    mol: Molecule,
    faces: object,
    *,
    role: FusionComponentRole | str | None = None,
    input_to_skeleton_atom: Mapping[int, int] | None = None,
) -> tuple[FusionComponentMatch, ...]:
    """Match faces using the process-wide component registry."""

    return fusion_component_registry().match_faces(
        mol,
        faces,
        role=role,
        input_to_skeleton_atom=input_to_skeleton_atom,
    )
