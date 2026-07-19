"""Graph-template model for retained fused parent names.

The production retained-fused recognizers are still being migrated.  This
module defines the audited data shape used by the next matcher: templates are
keyed by locants and graph structure, not by SMILES or SMARTS strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .molecule import Molecule
from .naming_data import load_json_table
from .nomenclature import RULES

ALLOWED_BOND_CLASSES = {"single", "double", "aromatic", "mancude", "fusion"}

NAPHTHALENOID_10_TEMPLATE = {
    "locants": ["1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"],
    "bonds": [
        {"locants": ["1", "2"]},
        {"locants": ["2", "3"]},
        {"locants": ["3", "4"]},
        {"locants": ["4", "4a"]},
        {"locants": ["4a", "8a"], "bond_class": "fusion"},
        {"locants": ["8a", "1"]},
        {"locants": ["4a", "5"]},
        {"locants": ["5", "6"]},
        {"locants": ["6", "7"]},
        {"locants": ["7", "8"]},
        {"locants": ["8", "8a"]},
    ],
    "rings": [
        ["1", "2", "3", "4", "4a", "8a"],
        ["4a", "5", "6", "7", "8", "8a"],
    ],
    "fusion_atoms": ["4a", "8a"],
    "peripheral_atoms": ["1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"],
}


def _shared_graph_template(
    locants: list[str],
    bonds: list[tuple[str, str]],
    fusion_atoms: list[str],
    mancude_double_bonds: int,
) -> dict:
    return {
        "locants": locants,
        "bonds": [
            {
                "locants": [left, right],
                **({"bond_class": "fusion"} if left in fusion_atoms and right in fusion_atoms else {}),
            }
            for left, right in bonds
        ],
        "fusion_atoms": fusion_atoms,
        "peripheral_atoms": locants,
        "mancude_double_bonds": mancude_double_bonds,
    }


LINEAR_TRICYCLIC_14_TEMPLATE = _shared_graph_template(
    ["1", "2", "3", "4", "4a", "5", "5a", "6", "7", "8", "9", "9a", "10", "10a"],
    [
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "4a"),
        ("4a", "5"),
        ("5", "5a"),
        ("5a", "6"),
        ("6", "7"),
        ("7", "8"),
        ("8", "9"),
        ("9", "9a"),
        ("5a", "9a"),
        ("9a", "10"),
        ("10", "10a"),
        ("10a", "1"),
        ("4a", "10a"),
    ],
    ["4a", "5a", "9a", "10a"],
    7,
)

PHENANTHRENOID_14_TEMPLATE = _shared_graph_template(
    ["1", "2", "3", "4", "4a", "5", "6", "6a", "7", "8", "9", "10", "10a", "10b"],
    [
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "4a"),
        ("4a", "5"),
        ("5", "6"),
        ("6", "6a"),
        ("6a", "7"),
        ("7", "8"),
        ("8", "9"),
        ("9", "10"),
        ("10", "10a"),
        ("6a", "10a"),
        ("10a", "10b"),
        ("10b", "1"),
        ("4a", "10b"),
    ],
    ["4a", "6a", "10a", "10b"],
    7,
)

ACRIDINOID_14_TEMPLATE = _shared_graph_template(
    ["1", "2", "3", "4", "4a", "10", "10a", "5", "6", "7", "8", "8a", "9", "9a"],
    [
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "4a"),
        ("4a", "10"),
        ("10", "10a"),
        ("10a", "5"),
        ("5", "6"),
        ("6", "7"),
        ("7", "8"),
        ("8", "8a"),
        ("10a", "8a"),
        ("8a", "9"),
        ("9", "9a"),
        ("9a", "1"),
        ("4a", "9a"),
    ],
    ["4a", "10a", "8a", "9a"],
    7,
)

CARBAZOLOID_13_TEMPLATE = _shared_graph_template(
    ["1", "2", "3", "4", "4a", "4b", "5", "6", "7", "8", "8a", "9", "9a"],
    [
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "4a"),
        ("4a", "4b"),
        ("4b", "5"),
        ("5", "6"),
        ("6", "7"),
        ("7", "8"),
        ("8", "8a"),
        ("4b", "8a"),
        ("8a", "9"),
        ("9", "9a"),
        ("9a", "1"),
        ("4a", "9a"),
    ],
    ["4a", "4b", "8a", "9a"],
    6,
)

PURINOID_9_TEMPLATE = _shared_graph_template(
    ["1", "2", "3", "4", "9", "8", "7", "5", "6"],
    [
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "9"),
        ("9", "8"),
        ("8", "7"),
        ("7", "5"),
        ("5", "4"),
        ("5", "6"),
        ("6", "1"),
    ],
    ["4", "5"],
    4,
)

INDAZOLOID_9_TEMPLATE = _shared_graph_template(
    ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"],
    [
        ("1", "2"),
        ("2", "3"),
        ("3", "3a"),
        ("3a", "4"),
        ("4", "5"),
        ("5", "6"),
        ("6", "7"),
        ("7", "7a"),
        ("7a", "1"),
        ("3a", "7a"),
    ],
    ["3a", "7a"],
    4,
)

RETAINED_FUSED_BASE_TEMPLATES = {
    "naphthalenoid_10": NAPHTHALENOID_10_TEMPLATE,
    "linear_tricyclic_14": LINEAR_TRICYCLIC_14_TEMPLATE,
    "phenanthrenoid_14": PHENANTHRENOID_14_TEMPLATE,
    "acridinoid_14": ACRIDINOID_14_TEMPLATE,
    "carbazoloid_13": CARBAZOLOID_13_TEMPLATE,
    "purinoid_9": PURINOID_9_TEMPLATE,
    "indazoloid_9": INDAZOLOID_9_TEMPLATE,
}


@dataclass(frozen=True)
class RetainedFusedAtomTemplate:
    locant: str
    symbol: str = "C"
    charge: int = 0
    aromatic: bool = True
    fusion: bool = False
    default_h: bool = False
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
    mancude_double_bonds: int | None = None

    @property
    def atom_by_locant(self) -> dict[str, RetainedFusedAtomTemplate]:
        return {atom.locant: atom for atom in self.atoms}


@dataclass(frozen=True)
class RetainedFusedTemplateMatch:
    template: RetainedFusedGraphTemplate
    atom_to_locant: dict[int, str]
    locant_to_atom: dict[str, int]
    matched_atoms: frozenset[int]
    indicated_h: tuple[str, ...]
    trace: tuple[str, ...] = ()


def retained_fused_graph_templates(*, include_disabled: bool = False) -> tuple[RetainedFusedGraphTemplate, ...]:
    """Return graph-template retained fused parent rows from the rule registry."""

    templates: list[RetainedFusedGraphTemplate] = []
    for row in load_json_table("retained_fused_graph_templates.json").get("parents", ()):
        template = retained_fused_template_from_data(row)
        if include_disabled or template.enabled:
            templates.append(template)
    for row in RULES.retained.fused_polycycle_specs:
        template_data = row.get("template")
        if template_data is None:
            continue
        template = retained_fused_template_from_data(row)
        if include_disabled or template.enabled:
            templates.append(template)
    return tuple(templates)


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

    validate_retained_fused_template(template)
    atom_set = set(atom_indices)
    if len(atom_set) != len(template.atoms):
        return None

    atom_by_locant = template.atom_by_locant
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
            if _atom_matches_template(mol, atom_idx, atom_by_locant[locant], allow_nonaromatic=allow_nonaromatic)
            and molecule_degrees[atom_idx] == template_degrees[locant]
        ]
        for locant in template.locants
    }
    if any(not values for values in candidates.values()):
        return None

    assignments = _match_locants_backtracking(
        mol,
        locants_by_constraint,
        candidates,
        template_neighbors,
    )
    if not assignments:
        return None

    assignment = assignments[0]
    return _template_match_from_assignment(template, atom_set, assignment)


def _match_all_retained_fused_template(
    mol: Molecule,
    atom_indices: set[int] | list[int] | tuple[int, ...],
    template: RetainedFusedGraphTemplate,
    *,
    allow_nonaromatic: bool = False,
) -> list[RetainedFusedTemplateMatch]:
    validate_retained_fused_template(template)
    atom_set = set(atom_indices)
    if len(atom_set) != len(template.atoms):
        return []

    atom_by_locant = template.atom_by_locant
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
            if _atom_matches_template(mol, atom_idx, atom_by_locant[locant], allow_nonaromatic=allow_nonaromatic)
            and molecule_degrees[atom_idx] == template_degrees[locant]
        ]
        for locant in template.locants
    }
    if any(not values for values in candidates.values()):
        return []

    assignments = _match_locants_backtracking(mol, locants_by_constraint, candidates, template_neighbors)
    return [_template_match_from_assignment(template, atom_set, assignment) for assignment in assignments]


def _template_match_from_assignment(
    template: RetainedFusedGraphTemplate,
    atom_set: set[int],
    assignment: dict[str, int],
) -> RetainedFusedTemplateMatch:
    locant_to_atom = {locant: assignment[locant] for locant in template.locants}
    atom_to_locant = {atom_idx: locant for locant, atom_idx in locant_to_atom.items()}
    return RetainedFusedTemplateMatch(
        template=template,
        atom_to_locant=atom_to_locant,
        locant_to_atom=locant_to_atom,
        matched_atoms=frozenset(atom_set),
        indicated_h=template.default_indicated_h,
        trace=(f"Matched retained fused template {template.name}.",),
    )


def match_retained_fused_templates(
    mol: Molecule,
    atom_indices: set[int] | list[int] | tuple[int, ...],
    *,
    include_disabled: bool = False,
    allow_nonaromatic: bool = False,
) -> list[RetainedFusedTemplateMatch]:
    """Return retained fused template matches ranked by retained priority."""

    matches = [
        match
        for template in retained_fused_graph_templates(include_disabled=include_disabled)
        for match in _match_all_retained_fused_template(
            mol,
            atom_indices,
            template,
            allow_nonaromatic=allow_nonaromatic,
        )
    ]
    return sorted(
        matches,
        key=_retained_fused_match_rank,
    )


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
        aliases=tuple(str(alias) for alias in row.get("aliases", template_data.get("aliases", ()))),
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
        derivative_production_enabled=bool(template_data.get("derivative_production_enabled", False)),
        mancude_double_bonds=(
            int(template_data["mancude_double_bonds"])
            if template_data.get("mancude_double_bonds") is not None
            else None
        ),
    )
    validate_retained_fused_template(template)
    return template


def _expand_template_data(template_data: dict[str, Any]) -> dict[str, Any]:
    base_name = template_data.get("base_template")
    if base_name is None:
        expanded = dict(template_data)
        return _expand_locant_atom_shorthand(expanded)
    base_template = RETAINED_FUSED_BASE_TEMPLATES.get(str(base_name))
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
    degrees = {locant: 0 for locant in template.locants}
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
    allow_nonaromatic: bool = False,
) -> bool:
    atom = mol.atoms[atom_idx]
    if atom.symbol != atom_template.symbol:
        return False
    if atom.charge != atom_template.charge:
        return False
    if (
        atom_template.aromatic
        and not atom.is_aromatic
        and not allow_nonaromatic
        and not _is_retained_oxo_site(mol, atom_idx)
    ):
        return False
    if atom_template.default_h and atom.explicit_h_count + atom.total_h_count <= 0:
        return False
    return True


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
    locant = locants[len(assignment)]
    for atom_idx in candidates[locant]:
        if atom_idx in used_atoms:
            continue
        if not _is_assignment_compatible(mol, locant, atom_idx, template_neighbors, assignment):
            continue
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
