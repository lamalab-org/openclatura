"""Graph-template model for retained fused parent names.

The production retained-fused recognizers are still being migrated.  This
module defines the audited data shape used by the next matcher: templates are
keyed by locants and graph structure, not by SMILES or SMARTS strings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache, lru_cache
from typing import Any

from .assembly_parts import RetainedParentMetadata
from .molecule import Molecule
from .naming_data import load_json_table
from .nomenclature import RULES
from .retained_name_policy import retained_parent_name_policy
from .rules import multipliers

ALLOWED_BOND_CLASSES = {"single", "double", "aromatic", "mancude", "fusion"}

# Single-bonded in every mancude parent, so never indicated-hydrogen capacity.
FIXED_SATURATED_RING_ELEMENTS = frozenset({"O", "S", "Se", "Te"})


@lru_cache(maxsize=1)
def retained_fused_base_templates() -> dict[str, dict[str, Any]]:
    """Return the shared retained-fused skeletons owned by the template table."""

    raw_templates = load_json_table("retained_fused_graph_templates.json").get("base_templates", {})
    if not isinstance(raw_templates, dict):
        raise ValueError("retained fused base_templates must be a mapping")
    templates: dict[str, dict[str, Any]] = {}
    for name, template in raw_templates.items():
        if not isinstance(template, dict):
            raise ValueError(f"Retained fused base template {name!r} must be a mapping.")
        templates[str(name)] = dict(template)
    return templates


@dataclass(frozen=True)
class RetainedFusedAtomTemplate:
    locant: str
    symbol: str = "C"
    charge: int = 0
    aromatic: bool = True
    fusion: bool = False
    default_h: bool = False
    saturated: bool = False
    interior: bool = False


@dataclass(frozen=True)
class RetainedFusedBondTemplate:
    locants: tuple[str, str]
    bond_class: str = "aromatic"


@dataclass(frozen=True)
class RetainedFusedGraphTemplate:
    name: str
    pin: bool
    priority: int
    aliases: tuple[str, ...]
    attached_prefix: str | None
    derivative_stem: str | None
    default_indicated_h: tuple[str, ...]
    locants: tuple[str, ...]
    atoms: tuple[RetainedFusedAtomTemplate, ...]
    bonds: tuple[RetainedFusedBondTemplate, ...]
    rings: tuple[tuple[str, ...], ...]
    fusion_atoms: tuple[str, ...]
    peripheral_atoms: tuple[str, ...]
    interior_atoms: tuple[str, ...]
    numbering_policy: str = "retained_template"
    aromatic_equivalence_policy: str = "neutral_kekule_equivalent"
    enabled: bool = False
    derivative_production_enabled: bool = False
    implied_stereo: bool = False
    mancude_double_bonds: int | None = None
    preferred_name: str | None = None

    @property
    def atom_by_locant(self) -> dict[str, RetainedFusedAtomTemplate]:
        return {atom.locant: atom for atom in self.atoms}

    @property
    def output_name(self) -> str:
        return self.preferred_name or self.name

    @property
    def indicated_hydrogen_count(self) -> int:
        """Indicated hydrogens this parent hydride supports.

        Declared carbon sites (9H-xanthene, 2H-pyran) when it has them, else the
        positions holding no mancude bond less the chalcogens, which are
        single-bonded anyway -- 1,4-benzodioxine's oxygens are not hydro sites.
        A bridgehead spends all three bonds inside the rings, so it has none
        left for a hydrogen: indolizine's N4 supports no indicated H.
        """

        if self.default_indicated_h:
            return len(self.default_indicated_h)
        return sum(
            1
            for atom in self.atoms
            if not atom.aromatic and not atom.fusion and atom.symbol not in FIXED_SATURATED_RING_ELEMENTS
        )


@dataclass(frozen=True)
class RetainedFusedTemplateMatch:
    template: RetainedFusedGraphTemplate
    atom_to_locant: dict[int, str]
    locant_to_atom: dict[str, int]
    matched_atoms: frozenset[int]
    indicated_h: tuple[str, ...]
    trace: tuple[str, ...] = ()


@lru_cache(maxsize=2)
def retained_fused_graph_templates(*, include_disabled: bool = False) -> tuple[RetainedFusedGraphTemplate, ...]:
    """Return graph-template retained fused parent rows from the rule registry."""

    templates: list[RetainedFusedGraphTemplate] = []
    parent_rows = list(load_json_table("retained_fused_graph_templates.json").get("parents", ()))
    parent_rows.extend(load_json_table("retained_fused_hydrocarbon_templates.json").get("parents", ()))
    for row in parent_rows:
        template = retained_fused_template_from_data(row)
        if include_disabled or template.enabled:
            templates.append(template)
    existing_names = {template.name for template in templates}
    generated_templates = (*_generated_acene_templates(), *_generated_polyaphene_templates())
    for template in generated_templates:
        if template.name not in existing_names and (include_disabled or template.enabled):
            templates.append(template)
    for row in RULES.retained.fused_polycycle_specs:
        template_data = row.get("template")
        if template_data is None:
            continue
        template = retained_fused_template_from_data(row)
        if include_disabled or template.enabled:
            templates.append(template)
    return tuple(templates)


@lru_cache(maxsize=1)
def _generated_acene_templates() -> tuple[RetainedFusedGraphTemplate, ...]:
    """Build higher linear-acene parents from the P-25.1.2 series table.

    Acenes with four or more linearly fused benzene rings share one locant-graph
    construction.  Keeping the member names and priorities in data lets the
    series grow without adding copied atom/bond templates or runtime structure
    strings.  Explicit historical templates take precedence above so existing
    audited rows, such as pentacene, remain unchanged.
    """

    series = load_json_table("retained_fused_series.json").get("acenes", {})
    minimum = int(series.get("minimum_ring_count", 4))
    maximum = int(series.get("maximum_ring_count", minimum))
    overrides = {int(count): dict(values) for count, values in series.get("member_overrides", {}).items()}
    rows = []
    for ring_count in range(minimum, maximum + 1):
        name = f"{multipliers.basic(ring_count)}cene"
        rows.append({"ring_count": ring_count, "name": name, **overrides.get(ring_count, {})})
    return tuple(_acene_template_from_data(row) for row in rows)


def _acene_template_from_data(row: dict[str, Any]) -> RetainedFusedGraphTemplate:
    ring_count = int(row["ring_count"])
    if ring_count < 4:
        raise ValueError("Generated acene templates require at least four fused rings.")

    locants, edges = _higher_acene_locant_graph(ring_count)
    fusion_atoms = tuple(locant for locant in locants if not locant.isdigit())
    name = str(row["name"])
    template = RetainedFusedGraphTemplate(
        name=name,
        pin=bool(row.get("pin", True)),
        priority=int(row.get("priority", 1000)),
        aliases=tuple(str(alias) for alias in row.get("aliases", ())),
        attached_prefix=str(row.get("attached_prefix", f"{name.removesuffix('e')}o")),
        derivative_stem=str(row.get("derivative_stem", name.removesuffix("e"))),
        default_indicated_h=(),
        locants=locants,
        atoms=tuple(RetainedFusedAtomTemplate(locant=locant, fusion=locant in fusion_atoms) for locant in locants),
        bonds=tuple(RetainedFusedBondTemplate(locants=edge) for edge in edges),
        rings=_smallest_ring_basis(locants, edges),
        fusion_atoms=fusion_atoms,
        peripheral_atoms=tuple(locant for locant in locants if locant.isdigit()),
        interior_atoms=(),
        numbering_policy="generated_acene_series",
        enabled=bool(row.get("enabled", True)),
        derivative_production_enabled=bool(row.get("derivative_production_enabled", True)),
        mancude_double_bonds=2 * ring_count + 1,
    )
    validate_retained_fused_template(template)
    return template


@lru_cache(maxsize=1)
def _generated_polyaphene_templates() -> tuple[RetainedFusedGraphTemplate, ...]:
    """Build the angular polyaphene series from P-25.1.2 family data.

    A polyaphene member has a perimeter cycle of ``4n + 2`` atoms and
    ``n - 1`` fusion chords.  Its angular bay moves deterministically as the
    family grows.  The construction below emits the standard fused locants as
    well as the topology, so derivatives can reuse the same atom-to-locant map
    instead of treating the family as parent-name-only aliases.
    """

    series = load_json_table("retained_fused_series.json").get("polyaphenes", {})
    minimum = int(series.get("minimum_ring_count", 5))
    maximum = int(series.get("maximum_ring_count", minimum))
    overrides = {int(count): dict(values) for count, values in series.get("member_overrides", {}).items()}
    rows = []
    for ring_count in range(minimum, maximum + 1):
        name = f"{multipliers.basic(ring_count)}phene"
        rows.append({"ring_count": ring_count, "name": name, **overrides.get(ring_count, {})})
    return tuple(_polyaphene_template_from_data(row) for row in rows)


def _polyaphene_template_from_data(row: dict[str, Any]) -> RetainedFusedGraphTemplate:
    ring_count = int(row["ring_count"])
    if ring_count < 4:
        raise ValueError("Generated polyaphene templates require at least four fused rings.")

    locants, edges = _polyaphene_locant_graph(ring_count)
    fusion_atoms = tuple(locant for locant in locants if not locant.isdigit())
    name = str(row["name"])
    template = RetainedFusedGraphTemplate(
        name=name,
        pin=bool(row.get("pin", True)),
        priority=int(row.get("priority", 1000)),
        aliases=tuple(str(alias) for alias in row.get("aliases", ())),
        attached_prefix=str(row.get("attached_prefix", f"{name.removesuffix('e')}o")),
        derivative_stem=str(row.get("derivative_stem", name.removesuffix("e"))),
        default_indicated_h=(),
        locants=locants,
        atoms=tuple(RetainedFusedAtomTemplate(locant=locant, fusion=locant in fusion_atoms) for locant in locants),
        bonds=tuple(RetainedFusedBondTemplate(locants=edge) for edge in edges),
        rings=_smallest_ring_basis(locants, edges),
        fusion_atoms=fusion_atoms,
        peripheral_atoms=tuple(locant for locant in locants if locant.isdigit()),
        interior_atoms=(),
        numbering_policy="generated_polyaphene_series",
        enabled=bool(row.get("enabled", True)),
        derivative_production_enabled=bool(row.get("derivative_production_enabled", True)),
        mancude_double_bonds=2 * ring_count + 1,
    )
    validate_retained_fused_template(template)
    return template


def _polyaphene_locant_graph(ring_count: int) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return the standard locanted graph for an angular polyaphene member."""

    maximum_numeric_locant = 2 * ring_count + 4
    omitted_first_side_fusion = (ring_count + 7) // 2
    first_side_numbers = tuple(
        value for value in range(4, ring_count + 4) if value != omitted_first_side_fusion
    )
    second_side_numbers = tuple(range(ring_count + 7, maximum_numeric_locant + 1))
    bay_number = (3 * ring_count) // 2 + 6

    locants: list[str] = []
    for value in range(1, maximum_numeric_locant + 1):
        locants.append(str(value))
        if value in first_side_numbers or value in second_side_numbers:
            locants.append(f"{value}a")
        if value == bay_number:
            locants.append(f"{value}b")

    edges = [(left, right) for left, right in zip(locants[:-1], locants[1:], strict=True)]
    edges.append((locants[-1], locants[0]))
    first_side = tuple(f"{value}a" for value in reversed(first_side_numbers))
    second_side = tuple(
        locant
        for locant in locants
        if locant == f"{bay_number}b" or locant in {f"{value}a" for value in second_side_numbers}
    )
    edges.extend(zip(first_side, second_side, strict=True))
    return tuple(locants), tuple(edges)


def _higher_acene_locant_graph(ring_count: int) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return the standard locanted graph for tetracene and higher acenes."""

    left_fusions = tuple(f"{value}a" for value in range(4, ring_count + 3))
    right_start = ring_count + 6
    right_fusions = tuple(f"{value}a" for value in range(right_start, 2 * ring_count + 5))

    perimeter: list[str] = ["1", "2", "3", "4", left_fusions[0]]
    for value in range(5, ring_count + 3):
        perimeter.extend((str(value), f"{value}a"))
    perimeter.extend(str(value) for value in range(ring_count + 3, right_start + 1))
    perimeter.append(right_fusions[0])
    for value in range(right_start + 1, 2 * ring_count + 5):
        perimeter.extend((str(value), f"{value}a"))

    edges = [(left, right) for left, right in zip(perimeter[:-1], perimeter[1:], strict=True)]
    edges.append((perimeter[-1], perimeter[0]))
    edges.extend(zip(reversed(left_fusions), right_fusions, strict=True))
    return tuple(perimeter), tuple(edges)


def _smallest_ring_basis(
    locants: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    *,
    maximum_ring_size: int = 8,
) -> tuple[tuple[str, ...], ...]:
    """Return a deterministic short-cycle basis for a locant graph.

    Retained fused parents are small and their elementary rings are normally
    five- or six-membered.  Enumerating bounded simple cycles and selecting an
    independent basis keeps generated registry rows self-contained without a
    graph-library dependency.
    """

    adjacency = {locant: set() for locant in locants}
    edge_keys = {frozenset(edge) for edge in edges}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    cycles: set[tuple[str, ...]] = set()
    order = {locant: index for index, locant in enumerate(locants)}
    for start in locants:
        stack: list[tuple[str, tuple[str, ...], frozenset[str]]] = [(start, (start,), frozenset({start}))]
        while stack:
            current, path, visited = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor == start and len(path) >= 3:
                    cycles.add(_canonical_cycle(path, order))
                    continue
                if neighbor in visited or order[neighbor] < order[start] or len(path) >= maximum_ring_size:
                    continue
                stack.append((neighbor, (*path, neighbor), visited | {neighbor}))

    edge_order = {edge: index for index, edge in enumerate(sorted(edge_keys, key=lambda item: sorted(item)))}
    ranked = sorted(cycles, key=lambda cycle: (len(cycle), tuple(order[locant] for locant in cycle)))
    target_rank = len(edges) - len(locants) + 1
    pivots: dict[int, int] = {}
    selected: list[tuple[str, ...]] = []
    for cycle in ranked:
        vector = 0
        for index, left in enumerate(cycle):
            right = cycle[(index + 1) % len(cycle)]
            vector ^= 1 << edge_order[frozenset((left, right))]
        reduced = vector
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = reduced
                selected.append(cycle)
                break
            reduced ^= pivots[pivot]
        if len(selected) == target_rank:
            return tuple(selected)
    raise ValueError("Could not construct a complete bounded ring basis for retained fused graph.")


def _canonical_cycle(cycle: tuple[str, ...], order: dict[str, int]) -> tuple[str, ...]:
    rotations = []
    for sequence in (cycle, tuple(reversed(cycle))):
        rotations.extend(sequence[index:] + sequence[:index] for index in range(len(sequence)))
    return min(rotations, key=lambda item: tuple(order[locant] for locant in item))


@cache
def retained_parent_metadata(parent_name: str) -> RetainedParentMetadata | None:
    """Return metadata only when the spelling identifies one exact hydride.

    Bare aliases such as ``isoindole`` do not identify the indicated-hydrogen
    tautomer.  Those sites must be derived from the selected molecular graph;
    production graph-template matches pass their metadata directly.
    """

    for template in retained_fused_graph_templates(include_disabled=True):
        if parent_name == template.name:
            return RetainedParentMetadata(
                default_indicated_h=template.default_indicated_h,
                fusion_locants=template.fusion_atoms,
                derivative_stem=template.derivative_stem,
                indicated_hydrogen_count=template.indicated_hydrogen_count,
                mancude_double_bonds=template.mancude_double_bonds or 0,
            )
    return None


def pending_retained_fused_parent_names() -> tuple[str, ...]:
    """Return retained fused candidates that still need graph templates.

    These rows are planning metadata only.  They are not considered by the
    matcher and therefore cannot affect production naming.
    """

    return tuple(
        str(row["name"]) for row in load_json_table("retained_fused_graph_templates.json").get("pending_parents", ())
    )


def match_retained_fused_template(
    mol: Molecule,
    atom_indices: set[int] | list[int] | tuple[int, ...],
    template: RetainedFusedGraphTemplate,
    *,
    allow_nonaromatic: bool = False,
) -> RetainedFusedTemplateMatch | None:
    """Match one retained fused graph template to a molecule atom set.

    It returns graph atom IDs bound to display locants and does not use SMILES
    or SMARTS.  If a template has several graph automorphisms, this returns the
    stable first match; production code should use
    :func:`match_retained_fused_templates` to keep all locant maps available to
    the numbering layer.
    """

    matches = _match_all_retained_fused_template(mol, atom_indices, template, allow_nonaromatic=allow_nonaromatic)
    return matches[0] if matches else None


def _ring_fusion_stereo_is_assigned(mol: Molecule, atom_set: set[int]) -> bool:
    """Whether every ring-fusion centre of a matched skeleton has a configuration.

    A steroid name carries the configuration of its ring fusions, so spelling a
    structure that leaves them open as ``gonane`` asserts stereochemistry the
    structure does not have.  Those fall back to the von Baeyer name.
    """

    for atom_idx in atom_set:
        atom = mol.atoms[atom_idx]
        ring_neighbors = sum(1 for neighbor in mol.get_neighbors(atom_idx) if neighbor in atom_set)
        if ring_neighbors < 3:
            continue
        if atom.stereo is None and atom.raw_stereo is None:
            return False
    return True


def _match_all_retained_fused_template(
    mol: Molecule,
    atom_indices: set[int] | list[int] | tuple[int, ...],
    template: RetainedFusedGraphTemplate,
    *,
    allow_nonaromatic: bool = False,
    allow_relocated_indicated_h: bool = False,
) -> list[RetainedFusedTemplateMatch]:
    atom_set = set(atom_indices)
    if len(atom_set) != len(template.atoms):
        return []
    if all(not atom.aromatic for atom in template.atoms) and any(
        not _is_saturated_site(mol, atom_idx, atom_set) for atom_idx in atom_set
    ):
        return []
    if template.implied_stereo and not _ring_fusion_stereo_is_assigned(mol, atom_set):
        return []

    atom_by_locant = _relocatable_atom_by_locant(template) if allow_relocated_indicated_h else template.atom_by_locant
    template_degrees = _template_degrees(template)
    molecule_degrees = {
        atom_idx: sum(1 for neighbor in mol.get_neighbors(atom_idx) if neighbor in atom_set) for atom_idx in atom_set
    }
    template_neighbors = _template_neighbors(template)
    locants_by_constraint = sorted(
        template.locants,
        key=lambda locant: (
            -template_degrees[locant],
            atom_by_locant[locant].symbol == "C",
            locant,
        ),
    )
    candidates = {
        locant: [
            atom_idx
            for atom_idx in atom_set
            if _atom_matches_template(
                mol, atom_idx, atom_by_locant[locant], ring_atoms=atom_set, allow_nonaromatic=allow_nonaromatic
            )
            and molecule_degrees[atom_idx] == template_degrees[locant]
        ]
        for locant in template.locants
    }
    if any(not values for values in candidates.values()):
        return []

    assignments = _match_locants_backtracking(mol, locants_by_constraint, candidates, template_neighbors)
    if not allow_relocated_indicated_h:
        derive = not template.default_indicated_h and template.indicated_hydrogen_count > 0
        return [
            _template_match_from_assignment(
                template,
                atom_set,
                assignment,
                indicated_h=(
                    _relocated_indicated_h(mol, template, atom_set, assignment, strict=True) if derive else None
                ),
            )
            for assignment in assignments
        ]
    relocated = [
        (assignment, indicated_h)
        for assignment in assignments
        if (indicated_h := _relocated_indicated_h(mol, template, atom_set, assignment)) is not None
    ]
    return [
        _template_match_from_assignment(template, atom_set, assignment, indicated_h=indicated_h)
        for assignment, indicated_h in relocated
    ]


def _relocatable_atom_by_locant(template: RetainedFusedGraphTemplate) -> dict[str, RetainedFusedAtomTemplate]:
    """The template with its indicated-H site free to move."""

    movable = {
        atom.symbol
        for atom in template.atoms
        if not atom.aromatic
        and not atom.fusion
        and atom.symbol != "C"
        and atom.symbol not in FIXED_SATURATED_RING_ELEMENTS
    }
    relocatable: dict[str, RetainedFusedAtomTemplate] = {}
    for locant, atom in template.atom_by_locant.items():
        if locant in template.default_indicated_h and atom.symbol == "C":
            relocatable[locant] = replace(atom, saturated=False, default_h=False)
        elif atom.symbol in movable and not atom.fusion:
            relocatable[locant] = replace(atom, aromatic=False, default_h=False)
        else:
            relocatable[locant] = atom
    return relocatable


def _relocated_indicated_h(
    mol: Molecule,
    template: RetainedFusedGraphTemplate,
    atom_set: set[int],
    assignment: dict[str, int],
    *,
    strict: bool = False,
) -> tuple[str, ...] | None:
    """Where the indicated hydrogen sits here, or None if the ring is too unsaturated."""

    saturated = tuple(
        locant
        for locant in template.locants
        if template.atom_by_locant[locant].symbol not in FIXED_SATURATED_RING_ELEMENTS
        and _is_saturated_site(mol, assignment[locant], atom_set)
    )
    needed = template.indicated_hydrogen_count
    if len(saturated) < needed:
        return None
    movable = [locant for locant in saturated if not _is_retained_oxo_site(mol, assignment[locant])]

    if len(movable) < needed:
        movable = list(saturated)
    if len(movable) < needed or (strict and len(movable) != needed):
        return None
    return tuple(sorted(movable, key=_locant_sort_key)[:needed])


def _template_match_from_assignment(
    template: RetainedFusedGraphTemplate,
    atom_set: set[int],
    assignment: dict[str, int],
    *,
    indicated_h: tuple[str, ...] | None = None,
) -> RetainedFusedTemplateMatch:
    locant_to_atom = {locant: assignment[locant] for locant in template.locants}
    atom_to_locant = {atom_idx: locant for locant, atom_idx in locant_to_atom.items()}
    return RetainedFusedTemplateMatch(
        template=template,
        atom_to_locant=atom_to_locant,
        locant_to_atom=locant_to_atom,
        matched_atoms=frozenset(atom_set),
        indicated_h=template.default_indicated_h if indicated_h is None else indicated_h,
        trace=(f"Matched retained fused template {template.name}.",),
    )


TopologyKey = tuple[
    int,
    tuple[tuple[str, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[tuple[int, int], int], ...],
]


@lru_cache(maxsize=2)
def _templates_by_topology(include_disabled: bool) -> dict[TopologyKey, tuple[RetainedFusedGraphTemplate, ...]]:
    """Index templates by cheap graph invariants before exact matching."""

    index: dict[TopologyKey, list[RetainedFusedGraphTemplate]] = {}
    for template in retained_fused_graph_templates(include_disabled=include_disabled):
        index.setdefault(_template_topology_key(template), []).append(template)
    return {key: tuple(templates) for key, templates in index.items()}


def _template_topology_key(template: RetainedFusedGraphTemplate) -> TopologyKey:
    degrees = _template_degrees(template)
    return _topology_key(
        (atom.symbol for atom in template.atoms),
        (degrees[locant] for locant in template.locants),
        ((degrees[left], degrees[right]) for left, right in (bond.locants for bond in template.bonds)),
    )


def retained_graph_template_topology_key(template: RetainedFusedGraphTemplate) -> TopologyKey:
    """Return the shared cheap topology key for a retained graph template."""

    return _template_topology_key(template)


def _molecule_topology_key(mol: Molecule, atom_set: set[int]) -> TopologyKey:
    degrees = {
        atom_idx: sum(neighbor in atom_set for neighbor in mol.get_neighbors(atom_idx)) for atom_idx in atom_set
    }
    return _topology_key(
        (mol.atoms[atom_idx].symbol for atom_idx in atom_set),
        degrees.values(),
        (
            (degrees[atom_idx], degrees[neighbor])
            for atom_idx in atom_set
            for neighbor in mol.get_neighbors(atom_idx)
            if neighbor in atom_set and atom_idx < neighbor
        ),
    )


def molecule_graph_topology_key(mol: Molecule, atom_set: set[int]) -> TopologyKey:
    """Return the shared cheap topology key for an induced molecule graph."""

    return _molecule_topology_key(mol, atom_set)


def _topology_key(symbols, degrees, edge_degrees) -> TopologyKey:
    symbol_counts: dict[str, int] = {}
    degree_counts: dict[int, int] = {}
    edge_degree_counts: dict[tuple[int, int], int] = {}
    size = 0
    for symbol in symbols:
        size += 1
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    for degree in degrees:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
    for left_degree, right_degree in edge_degrees:
        degree_pair = tuple(sorted((left_degree, right_degree)))
        edge_degree_counts[degree_pair] = edge_degree_counts.get(degree_pair, 0) + 1
    return (
        size,
        tuple(sorted(symbol_counts.items())),
        tuple(sorted(degree_counts.items())),
        tuple(sorted(edge_degree_counts.items())),
    )


def match_retained_fused_templates(
    mol: Molecule,
    atom_indices: set[int] | list[int] | tuple[int, ...],
    *,
    include_disabled: bool = False,
    allow_nonaromatic: bool = False,
    allow_relocated_indicated_h: bool = False,
) -> list[RetainedFusedTemplateMatch]:
    """Return retained fused template matches ranked by retained priority."""

    atom_set = set(atom_indices)
    cache_key = (
        frozenset(atom_set),
        include_disabled,
        allow_nonaromatic,
        allow_relocated_indicated_h,
    )
    cached = mol._retained_fused_cache.get(cache_key)
    if cached is not None:
        return list(cached)
    candidates = _templates_by_topology(include_disabled).get(_molecule_topology_key(mol, atom_set), ())
    matches = [
        match
        for template in candidates
        for match in _match_all_retained_fused_template(
            mol,
            atom_set,
            template,
            allow_nonaromatic=allow_nonaromatic,
            allow_relocated_indicated_h=allow_relocated_indicated_h,
        )
    ]
    ranked = sorted(
        matches,
        key=_retained_fused_match_rank,
    )
    mol._retained_fused_cache[cache_key] = tuple(ranked)
    return ranked


def _retained_fused_match_rank(match: RetainedFusedTemplateMatch) -> tuple:
    """Rank retained fused matches by retained-parent and numbering criteria."""

    template = match.template
    hetero_locants = tuple(_locant_sort_key(atom.locant) for atom in template.atoms if atom.symbol != "C")
    fusion_locants = tuple(_locant_sort_key(locant) for locant in template.fusion_atoms)
    indicated_h_rank = tuple(_locant_sort_key(locant) for locant in match.indicated_h)
    atom_order = tuple(match.locant_to_atom[locant] for locant in template.locants)
    return (
        template.priority,
        indicated_h_rank,
        hetero_locants,
        fusion_locants,
        template.name,
        atom_order,
    )


def _locant_sort_key(locant: str) -> tuple[int, str]:
    digits = ""
    suffix = ""
    for char in locant:
        if char.isdigit() and not suffix:
            digits += char
        else:
            suffix += char
    return (int(digits) if digits else 10_000, suffix)


def retained_fused_template_from_data(row: dict[str, Any]) -> RetainedFusedGraphTemplate:
    """Parse and validate one retained fused graph-template row."""

    template_data = row.get("template")
    if not isinstance(template_data, dict):
        raise ValueError(f"Retained fused parent {row.get('name')!r} has no graph template.")
    template_data = _expand_template_data(template_data)

    name = str(row["name"])
    name_policy = retained_parent_name_policy(name)
    locants = tuple(str(locant) for locant in template_data.get("locants", row.get("locants", ())))
    atoms = tuple(_atom_template(item) for item in template_data.get("atoms", ()))
    bonds = tuple(_bond_template(item) for item in template_data.get("bonds", ()))
    rings = tuple(tuple(str(locant) for locant in ring) for ring in template_data.get("rings", ()))
    fusion_atoms = tuple(str(locant) for locant in template_data.get("fusion_atoms", ()))
    peripheral_atoms = tuple(str(locant) for locant in template_data.get("peripheral_atoms", locants))
    interior_atoms = tuple(str(locant) for locant in template_data.get("interior_atoms", ()))

    template = RetainedFusedGraphTemplate(
        name=name,
        pin=bool(row.get("pin", template_data.get("pin", True))),
        priority=int(row.get("priority", template_data.get("priority", 1000))),
        aliases=tuple(
            dict.fromkeys(
                (
                    *(str(alias) for alias in row.get("aliases", template_data.get("aliases", ()))),
                    *(
                        alias
                        for alias in (name_policy.accepted_aliases if name_policy is not None else ())
                        if alias != name
                    ),
                )
            )
        ),
        attached_prefix=row.get("attached_prefix", row.get("fusion_prefix", template_data.get("attached_prefix"))),
        derivative_stem=row.get("derivative_stem", template_data.get("derivative_stem")),
        default_indicated_h=tuple(
            str(locant) for locant in row.get("default_indicated_h", template_data.get("default_indicated_h", ()))
        ),
        locants=locants,
        atoms=atoms,
        bonds=bonds,
        rings=rings,
        fusion_atoms=fusion_atoms,
        peripheral_atoms=peripheral_atoms,
        interior_atoms=interior_atoms,
        numbering_policy=str(template_data.get("numbering_policy", "retained_template")),
        aromatic_equivalence_policy=str(template_data.get("aromatic_equivalence_policy", "neutral_kekule_equivalent")),
        enabled=bool(template_data.get("enabled", row.get("template_enabled", False))),
        implied_stereo=bool(template_data.get("implied_stereo", False)),
        derivative_production_enabled=bool(template_data.get("derivative_production_enabled", False)),
        mancude_double_bonds=(
            int(template_data["mancude_double_bonds"])
            if template_data.get("mancude_double_bonds") is not None
            else None
        ),
        preferred_name=name_policy.preferred_name if name_policy is not None else None,
    )
    validate_retained_fused_template(template)
    return template


def _expand_template_data(template_data: dict[str, Any]) -> dict[str, Any]:
    base_name = template_data.get("base_template")
    if base_name is None:
        expanded = dict(template_data)
        return _expand_locant_atom_shorthand(expanded)
    base_template = retained_fused_base_templates().get(str(base_name))
    if base_template is None:
        raise ValueError(f"Unknown retained fused base template {base_name!r}.")

    expanded = {**base_template, **template_data}
    return _expand_locant_atom_shorthand(expanded)


def _expand_locant_atom_shorthand(template_data: dict[str, Any]) -> dict[str, Any]:
    """Expand compact locant-keyed atom declarations.

    Many retained fused parents differ only by heteroatom and indicated-H
    locants over a declared skeleton.  Keeping that in data avoids copy/pasted
    atom arrays while still making the graph template explicit.
    """

    expanded = dict(template_data)
    if expanded.get("atoms"):
        return expanded
    if not expanded.get("locants"):
        return expanded

    heteroatoms = {str(item["locant"]): str(item["symbol"]) for item in template_data.get("heteroatoms", ())}
    atom_overrides = {str(item["locant"]): dict(item) for item in template_data.get("atom_overrides", ())}
    fusion_atoms = set(str(locant) for locant in expanded.get("fusion_atoms", ()))
    indicated_h = set(str(locant) for locant in expanded.get("default_indicated_h", ()))
    atoms = []
    for locant in expanded["locants"]:
        atom = {
            "locant": locant,
            "symbol": heteroatoms.get(locant, "C"),
            "fusion": locant in fusion_atoms,
            "default_h": locant in indicated_h,
        }
        atom.update(atom_overrides.get(locant, {}))
        atoms.append(atom)
    expanded["atoms"] = atoms
    return expanded


def validate_retained_fused_template(template: RetainedFusedGraphTemplate) -> None:
    """Validate internal consistency for one retained fused graph template."""

    if not template.name:
        raise ValueError("Retained fused template requires a name.")
    if not template.locants:
        raise ValueError(f"Retained fused template {template.name!r} has no locants.")
    if len(set(template.locants)) != len(template.locants):
        raise ValueError(f"Retained fused template {template.name!r} has duplicate locants.")
    if len({atom.locant for atom in template.atoms}) != len(template.atoms):
        raise ValueError(f"Retained fused template {template.name!r} has duplicate atom locants.")
    if {atom.locant for atom in template.atoms} != set(template.locants):
        raise ValueError(f"Retained fused template {template.name!r} atom locants do not match locant list.")

    locant_set = set(template.locants)
    for ring in template.rings:
        if len(ring) < 3:
            raise ValueError(f"Retained fused template {template.name!r} has a ring with fewer than 3 atoms.")
        missing = set(ring) - locant_set
        if missing:
            raise ValueError(f"Retained fused template {template.name!r} ring references unknown locants {missing}.")

    for atom in template.atoms:
        if atom.fusion and atom.locant not in template.fusion_atoms:
            raise ValueError(f"Fusion atom {atom.locant!r} is not listed in fusion_atoms for {template.name!r}.")
    if set(template.fusion_atoms) - locant_set:
        raise ValueError(f"Retained fused template {template.name!r} has unknown fusion atoms.")
    if set(template.peripheral_atoms) - locant_set:
        raise ValueError(f"Retained fused template {template.name!r} has unknown peripheral atoms.")
    if set(template.interior_atoms) - locant_set:
        raise ValueError(f"Retained fused template {template.name!r} has unknown interior atoms.")
    if set(template.default_indicated_h) - locant_set:
        raise ValueError(f"Retained fused template {template.name!r} has unknown indicated-H locants.")

    seen_bonds: set[tuple[str, str]] = set()
    for bond in template.bonds:
        if bond.bond_class not in ALLOWED_BOND_CLASSES:
            raise ValueError(f"Unknown bond class {bond.bond_class!r} in retained fused template {template.name!r}.")
        if len(bond.locants) != 2 or bond.locants[0] == bond.locants[1]:
            raise ValueError(f"Invalid bond locants {bond.locants!r} in retained fused template {template.name!r}.")
        if set(bond.locants) - locant_set:
            raise ValueError(f"Retained fused template {template.name!r} bond references unknown locants.")
        key = tuple(sorted(bond.locants))
        if key in seen_bonds:
            raise ValueError(f"Retained fused template {template.name!r} has duplicate bond {key!r}.")
        seen_bonds.add(key)


def template_molecule(template: RetainedFusedGraphTemplate) -> Molecule:
    """Build a local molecule graph from a retained fused template."""

    validate_retained_fused_template(template)
    mol = Molecule()
    locant_to_idx = {locant: idx for idx, locant in enumerate(template.locants)}
    atom_by_locant = template.atom_by_locant
    for locant in template.locants:
        atom = atom_by_locant[locant]
        mol.add_atom(
            atom.symbol,
            locant_to_idx[locant],
            charge=atom.charge,
            is_aromatic=atom.aromatic,
            explicit_h_count=1 if atom.default_h else 0,
            total_h_count=1 if atom.default_h else 0,
        )
    for idx, bond in enumerate(template.bonds):
        order = _bond_order(bond.bond_class)
        mol.add_bond(locant_to_idx[bond.locants[0]], locant_to_idx[bond.locants[1]], order=order, idx=idx)
    return mol


def _template_degrees(template: RetainedFusedGraphTemplate) -> dict[str, int]:
    degrees = dict.fromkeys(template.locants, 0)
    for bond in template.bonds:
        degrees[bond.locants[0]] += 1
        degrees[bond.locants[1]] += 1
    return degrees


def _template_neighbors(template: RetainedFusedGraphTemplate) -> dict[str, set[str]]:
    neighbors = {locant: set() for locant in template.locants}
    for bond in template.bonds:
        a, b = bond.locants
        neighbors[a].add(b)
        neighbors[b].add(a)
    return neighbors


def _atom_matches_template(
    mol: Molecule,
    atom_idx: int,
    atom_template: RetainedFusedAtomTemplate,
    *,
    ring_atoms: frozenset[int] | set[int] | None = None,
    allow_nonaromatic: bool = False,
) -> bool:
    atom = mol.atoms[atom_idx]
    if atom.symbol != atom_template.symbol:
        return False
    # Imidazolium is imidazole's ring; the ionic layer names the charge.
    if atom_template.charge and atom.charge != atom_template.charge:
        return False

    # Read as bond order, not RDKit aromaticity, so Kekule input matches too.
    if (
        atom_template.aromatic
        and not atom.is_aromatic
        and _is_saturated_site(mol, atom_idx, ring_atoms)
        and not allow_nonaromatic
        and not _is_retained_oxo_site(mol, atom_idx)
    ):
        return False
    if _cumulated_ring_site(mol, atom_idx, ring_atoms):
        return False
    if atom_template.default_h and atom.explicit_h_count + atom.total_h_count <= 0:
        return False
    # Tells 2H- from 4H-1-benzopyran: the position must really be saturated.
    if atom_template.saturated and not _is_saturated_site(mol, atom_idx, ring_atoms):
        return False
    return True


def _cumulated_ring_site(mol: Molecule, atom_idx: int, ring_atoms=None) -> bool:
    """Whether the atom carries two ring double bonds, as in a cumulated triene."""

    doubles = sum(
        1
        for neighbor in mol.get_neighbors(atom_idx)
        if (ring_atoms is None or neighbor in ring_atoms)
        and (bond := mol.get_bond(atom_idx, neighbor)) is not None
        and bond.order == 2
    )
    return doubles > 1


def _is_saturated_site(mol: Molecule, atom_idx: int, ring_atoms=None) -> bool:
    """No multiple bond inside the ring; an exocyclic =O does not unsaturate."""

    return all(
        bond.order == 1
        for neighbor in mol.get_neighbors(atom_idx)
        if (ring_atoms is None or neighbor in ring_atoms) and (bond := mol.get_bond(atom_idx, neighbor)) is not None
    )


def _is_retained_oxo_site(mol: Molecule, atom_idx: int) -> bool:
    """Return whether aromaticity was lost only because this carbon bears =O."""

    if mol.atoms[atom_idx].symbol != "C":
        return False
    return any(
        mol.atoms[neighbor].symbol == "O" and (bond := mol.get_bond(atom_idx, neighbor)) is not None and bond.order == 2
        for neighbor in mol.get_neighbors(atom_idx)
    )


def _match_locants_backtracking(
    mol: Molecule,
    locants: list[str],
    candidates: dict[str, list[int]],
    template_neighbors: dict[str, set[str]],
    *,
    max_matches: int = 256,
) -> list[dict[str, int]]:
    matches: list[dict[str, int]] = []
    _collect_locant_matches(
        mol,
        locants,
        candidates,
        template_neighbors,
        {},
        set(),
        matches,
        max_matches,
    )
    return sorted(matches, key=lambda assignment: tuple(assignment[locant] for locant in locants))


def _collect_locant_matches(
    mol: Molecule,
    locants: list[str],
    candidates: dict[str, list[int]],
    template_neighbors: dict[str, set[str]],
    assignment: dict[str, int],
    used_atoms: set[int],
    matches: list[dict[str, int]],
    max_matches: int,
) -> None:
    if len(matches) >= max_matches:
        return
    if len(assignment) == len(locants):
        matches.append(dict(assignment))
        return

    # Select the most constrained remaining locant after every assignment.
    # A fixed degree-sorted order leaves long symmetric fused hydrocarbons with
    # many disconnected partial permutations.  Dynamic frontier selection is
    # the same exact isomorphism search, but it grows outward from established
    # edges and rejects impossible mappings immediately.
    options_by_locant: list[tuple[int, int, int, str, list[int]]] = []
    for locant in locants:
        if locant in assignment:
            continue
        options = [
            atom_idx
            for atom_idx in candidates[locant]
            if atom_idx not in used_atoms
            and _is_assignment_compatible(mol, locant, atom_idx, template_neighbors, assignment)
        ]
        if not options:
            return
        assigned_neighbors = sum(neighbor in assignment for neighbor in template_neighbors[locant])
        options_by_locant.append((len(options), -assigned_neighbors, -len(template_neighbors[locant]), locant, options))
    _, _, _, locant, options = min(options_by_locant)
    for atom_idx in options:
        assignment[locant] = atom_idx
        used_atoms.add(atom_idx)
        _collect_locant_matches(
            mol,
            locants,
            candidates,
            template_neighbors,
            assignment,
            used_atoms,
            matches,
            max_matches,
        )
        used_atoms.remove(atom_idx)
        del assignment[locant]


def _is_assignment_compatible(
    mol: Molecule,
    locant: str,
    atom_idx: int,
    template_neighbors: dict[str, set[str]],
    assignment: dict[str, int],
) -> bool:
    for assigned_locant, assigned_atom in assignment.items():
        expected_bond = assigned_locant in template_neighbors[locant]
        actual_bond = mol.get_bond(atom_idx, assigned_atom) is not None
        if expected_bond != actual_bond:
            return False
    return True


def _atom_template(data: dict[str, Any]) -> RetainedFusedAtomTemplate:
    return RetainedFusedAtomTemplate(
        locant=str(data["locant"]),
        symbol=str(data.get("symbol", "C")),
        charge=int(data.get("charge", 0)),
        aromatic=bool(data.get("aromatic", True)),
        fusion=bool(data.get("fusion", False)),
        default_h=bool(data.get("default_h", False)),
        saturated=bool(data.get("saturated", False)),
        interior=bool(data.get("interior", False)),
    )


def _bond_template(data: dict[str, Any]) -> RetainedFusedBondTemplate:
    locants = data.get("locants")
    if not isinstance(locants, (list, tuple)):
        raise ValueError("Retained fused bond template requires a locants list.")
    return RetainedFusedBondTemplate(
        locants=(str(locants[0]), str(locants[1])),
        bond_class=str(data.get("bond_class", "aromatic")),
    )


def _bond_order(bond_class: str) -> int:
    if bond_class == "double":
        return 2
    return 1
