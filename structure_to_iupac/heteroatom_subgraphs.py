"""Heteroatom-starting recursive substituent naming."""

from collections.abc import Callable

from .formatting import (
    count_names,
    format_counted_prefixes,
    format_element_substituent,
    is_complex_prefix,
    strip_outer_parentheses,
)
from .molecule import Molecule
from .namer_config import (
    ALKYL_OXY_PREFIXES,
    HALOGEN_LAMBDA_SUFFIXES,
    HALOGEN_PREFIXES,
    SIMPLE_SELANYL_PREFIXES,
    SIMPLE_SULFANYL_PREFIXES,
)

BranchNamer = Callable[[Molecule, int, set[int], int | None], str]


def upstream_bond_order(mol: Molecule, start_idx: int, upstream_atom: int | None) -> int:
    if upstream_atom is None:
        return 0
    bond = mol.get_bond(start_idx, upstream_atom)
    return bond.order if bond else 0


def subgraph_neighbors(
    mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None, extra_exclude=None
) -> list[int]:
    blocked = set(extra_exclude or [])
    return [
        n
        for n in mol.get_neighbors(start_idx)
        if n not in exclude_atoms and n != upstream_atom and n not in blocked
    ]


def double_bonded_neighbors(mol: Molecule, center_idx: int, symbol: str) -> list[int]:
    return [
        n
        for n in mol.get_neighbors(center_idx)
        if mol.atoms[n].symbol == symbol and mol.get_bond(center_idx, n).order == 2
    ]


def first_substituent_neighbor(mol: Molecule, center_idx: int, excluded: set[int], require_carbon=False) -> int | None:
    for n in mol.get_neighbors(center_idx):
        if n in excluded:
            continue
        if require_carbon and not mol.atoms[n].is_carbon:
            continue
        return n
    return None


def stereo_prefix(atom) -> str:
    return f"({atom.stereo})-" if atom.stereo else ""


def sulfur_oxo_suffix(oxo_count: int) -> str:
    return "sulfonyl" if oxo_count == 2 else "sulfinyl"


def name_branch_or_none(
    mol: Molecule,
    branch_idx: int | None,
    exclude_atoms: set[int],
    upstream_atom: int,
    branch_namer: BranchNamer,
) -> str:
    if branch_idx is None:
        return ""
    return strip_outer_parentheses(branch_namer(mol, branch_idx, exclude_atoms, upstream_atom))


def name_carbonyl_like_fragment(
    mol: Molecule,
    center_idx: int,
    attach_idx: int,
    double_atoms: list[int],
    exclude_atoms: set[int],
    branch_suffix: str,
    fallback: str,
    branch_namer: BranchNamer,
    amino_base: str | None = None,
    wrap_result: bool = False,
) -> str:
    local_exclude = exclude_atoms | {attach_idx, center_idx} | set(double_atoms)
    branch_idx = first_substituent_neighbor(mol, center_idx, {attach_idx, *double_atoms}, require_carbon=True)
    if branch_idx is None:
        branch_idx = first_substituent_neighbor(mol, center_idx, {attach_idx, *double_atoms})

    branch = name_branch_or_none(mol, branch_idx, local_exclude, upstream_atom=center_idx, branch_namer=branch_namer)
    if not branch:
        return fallback
    if amino_base and branch.endswith("amino"):
        return f"({branch[:-5]}{amino_base})"
    result = f"{branch}{branch_suffix}"
    return f"({result})" if wrap_result else result


def format_amino_from_branches(branches: list[str], is_double: bool) -> str:
    if is_double:
        if len(branches) == 1:
            branch = strip_outer_parentheses(branches[0])
            if is_complex_prefix(branch):
                return f"(({branch})imino)"
            return f"({branch}imino)"
        return "imino"

    counts = count_names(branches)
    if len(counts) == 1 and list(counts.values())[0] == 1:
        branch = strip_outer_parentheses(branches[0])
        if branch.endswith(("carbonyl", "sulfonyl", "sulfinyl", "carbonothioyl")):
            return f"{branch}amino"
        if is_complex_prefix(branch):
            return f"(({branch})amino)"
        return f"({branch}amino)"

    return f"({format_counted_prefixes(branches)}amino)"


def format_lambda_substituent(
    mol: Molecule,
    start_idx: int,
    branches: list[str],
    stereo_prefix_text: str,
    base_suffix: str,
) -> str:
    valence = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))
    return f"({stereo_prefix_text}{format_counted_prefixes(branches)}-lambda^{valence}-{base_suffix})"


def name_oxygen_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str:
    is_double = upstream_bond_order(mol, start_idx, upstream_atom) == 2
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    start_atom = mol.atoms[start_idx]
    if not next_atoms:
        if is_double:
            return "oxo"
        if start_atom.charge == -1:
            return "oxido"
        return "hydroxy"

    nxt = next_atoms[0]
    s_oxygens = double_bonded_neighbors(mol, nxt, "O")
    if mol.atoms[nxt].symbol == "S" and s_oxygens:
        branch_idx = first_substituent_neighbor(mol, nxt, {start_idx, *s_oxygens})
        branch = name_branch_or_none(
            mol,
            branch_idx,
            exclude_atoms | {start_idx, nxt} | set(s_oxygens),
            nxt,
            branch_namer,
        )
        if branch:
            return f"({stereo_prefix(mol.atoms[nxt])}{branch}{sulfur_oxo_suffix(len(s_oxygens))}oxy)"
        return "sulfooxy"

    c_oxygens = double_bonded_neighbors(mol, nxt, "O")
    c_sulfurs = double_bonded_neighbors(mol, nxt, "S")
    if mol.atoms[nxt].is_carbon and len(c_oxygens) == 1:
        return name_carbonyl_like_fragment(
            mol, nxt, start_idx, c_oxygens, exclude_atoms, "carbonyloxy", "formyloxy", branch_namer, wrap_result=True
        )
    if mol.atoms[nxt].is_carbon and len(c_sulfurs) == 1:
        return name_carbonyl_like_fragment(
            mol,
            nxt,
            start_idx,
            c_sulfurs,
            exclude_atoms,
            "carbonothioyloxy",
            "methanethioyloxy",
            branch_namer,
            wrap_result=True,
        )
    if mol.atoms[nxt].symbol == "O":
        branch_idx = first_substituent_neighbor(mol, nxt, {start_idx})
        branch = name_branch_or_none(mol, branch_idx, exclude_atoms | {start_idx, nxt}, nxt, branch_namer)
        return f"({branch}peroxy)" if branch else "hydroperoxy"

    branch = branch_namer(mol, nxt, exclude_atoms | {start_idx}, start_idx)
    if branch:
        if branch in ALKYL_OXY_PREFIXES:
            return ALKYL_OXY_PREFIXES[branch]
        branch = strip_outer_parentheses(branch)
        if is_complex_prefix(branch):
            return f"(({branch})oxy)"
        return f"({branch}oxy)"
    return "hydroxy"


def name_nitrogen_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str:
    upstream_order = upstream_bond_order(mol, start_idx, upstream_atom)
    is_double = upstream_order == 2
    is_triple = upstream_order == 3
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    if not next_atoms:
        if is_double:
            return "imino"
        if is_triple:
            return "nitrilo"
        return "amino"

    branches = []
    for nxt in next_atoms:
        c_oxygens = double_bonded_neighbors(mol, nxt, "O")
        c_sulfurs = double_bonded_neighbors(mol, nxt, "S")
        if mol.atoms[nxt].is_carbon and len(c_oxygens) == 1:
            branches.append(
                name_carbonyl_like_fragment(
                    mol, nxt, start_idx, c_oxygens, exclude_atoms, "carbonyl", "formyl", branch_namer, amino_base="carbamoyl"
                )
            )
        elif mol.atoms[nxt].is_carbon and len(c_sulfurs) == 1:
            branches.append(
                name_carbonyl_like_fragment(
                    mol,
                    nxt,
                    start_idx,
                    c_sulfurs,
                    exclude_atoms,
                    "carbonothioyl",
                    "methanethioyl",
                    branch_namer,
                    amino_base="carbamothioyl",
                )
            )
        else:
            s_oxygens = double_bonded_neighbors(mol, nxt, "O")
            if mol.atoms[nxt].symbol == "S" and s_oxygens:
                branch_idx = first_substituent_neighbor(mol, nxt, {start_idx, *s_oxygens})
                branch = name_branch_or_none(
                    mol, branch_idx, exclude_atoms | {start_idx, nxt} | set(s_oxygens), nxt, branch_namer
                )
                if branch:
                    branches.append(f"{stereo_prefix(mol.atoms[nxt])}{branch}{sulfur_oxo_suffix(len(s_oxygens))}")
                else:
                    branches.append("sulfo")
            else:
                branch = branch_namer(mol, nxt, exclude_atoms | {start_idx}, start_idx)
                if branch:
                    branches.append(branch)

    return format_amino_from_branches(branches, is_double)


def name_sulfur_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str:
    is_double = upstream_bond_order(mol, start_idx, upstream_atom) == 2
    s_oxygens = double_bonded_neighbors(mol, start_idx, "O")
    s_nitrogens = double_bonded_neighbors(mol, start_idx, "N")
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom, s_oxygens + s_nitrogens)
    stereo_prefix_text = stereo_prefix(mol.atoms[start_idx])

    if len(s_oxygens) == 1 and len(s_nitrogens) == 1:
        suffix = "sulfonimidoyl" + ("idene" if is_double else "")
        if not next_atoms:
            return f"{stereo_prefix_text}{suffix}"
        if len(next_atoms) == 1:
            branch = name_branch_or_none(
                mol, next_atoms[0], exclude_atoms | {start_idx} | set(s_oxygens) | set(s_nitrogens), start_idx, branch_namer
            )
            return f"({stereo_prefix_text}{branch}{suffix})" if branch else f"{stereo_prefix_text}{suffix}"
        branches = [
            br
            for nxt in next_atoms
            if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx} | set(s_oxygens) | set(s_nitrogens), start_idx))
        ]
        branches.extend(["oxo"] * len(s_oxygens))
        branches.extend(["imino"] * len(s_nitrogens))
        return format_lambda_substituent(
            mol, start_idx, branches, stereo_prefix_text, "sulfanylidene" if is_double else "sulfanyl"
        )

    if s_oxygens:
        suffix = sulfur_oxo_suffix(len(s_oxygens)) + ("idene" if is_double else "")
        if not next_atoms:
            return f"{stereo_prefix_text}{suffix}"
        if len(next_atoms) == 1:
            branch = name_branch_or_none(mol, next_atoms[0], exclude_atoms | {start_idx} | set(s_oxygens), start_idx, branch_namer)
            return f"({stereo_prefix_text}{branch}{suffix})" if branch else f"{stereo_prefix_text}{suffix}"
        branches = [
            br for nxt in next_atoms if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx} | set(s_oxygens), start_idx))
        ]
        branches.extend(["oxo"] * len(s_oxygens))
        return format_lambda_substituent(
            mol, start_idx, branches, stereo_prefix_text, "sulfanylidene" if is_double else "sulfanyl"
        )

    if not next_atoms:
        return "thioxo" if is_double else f"{stereo_prefix_text}sulfanyl"

    if len(next_atoms) == 1:
        branch = branch_namer(mol, next_atoms[0], exclude_atoms | {start_idx}, start_idx)
        if branch in SIMPLE_SULFANYL_PREFIXES:
            return f"({stereo_prefix_text}{branch}sulfanyl)"
        if branch:
            return format_element_substituent(stereo_prefix_text, branch, "sulfanyl", is_double=is_double)
        return f"{stereo_prefix_text}{'sulfanylidene' if is_double else 'sulfanyl'}"

    branches = [br for nxt in next_atoms if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx}, start_idx))]
    return format_lambda_substituent(
        mol, start_idx, branches, stereo_prefix_text, "sulfanylidene" if is_double else "sulfanyl"
    )


def name_chalcogen_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    simple_prefixes: set[str],
    element_suffix: str,
    oxo_prefix: str,
    branch_namer: BranchNamer,
) -> str:
    is_double = upstream_bond_order(mol, start_idx, upstream_atom) == 2
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    stereo_prefix_text = stereo_prefix(mol.atoms[start_idx])
    if not next_atoms:
        return oxo_prefix if is_double else f"{stereo_prefix_text}{element_suffix}"
    if len(next_atoms) == 1:
        branch = branch_namer(mol, next_atoms[0], exclude_atoms | {start_idx}, start_idx)
        if branch in simple_prefixes:
            return f"({stereo_prefix_text}{branch}{element_suffix})"
        if branch:
            return format_element_substituent(stereo_prefix_text, branch, element_suffix, is_double=is_double)
        return f"{stereo_prefix_text}{element_suffix + ('idene' if is_double else '')}"
    branches = [br for nxt in next_atoms if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx}, start_idx))]
    return format_lambda_substituent(
        mol, start_idx, branches, stereo_prefix_text, element_suffix + ("idene" if is_double else "")
    )


def name_phosphorus_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str:
    is_double = upstream_bond_order(mol, start_idx, upstream_atom) == 2
    p_oxygens = double_bonded_neighbors(mol, start_idx, "O")
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom, p_oxygens)
    stereo_prefix_text = stereo_prefix(mol.atoms[start_idx])
    suffix = ("phosphoryl" if p_oxygens else "phosphanyl") + ("idene" if is_double else "")
    if not next_atoms:
        return f"{stereo_prefix_text}{suffix}"
    branches = [br for nxt in next_atoms if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx} | set(p_oxygens), start_idx))]
    return f"({stereo_prefix_text}{format_counted_prefixes(branches)}{suffix})"


def name_group_13_14_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str:
    is_double = upstream_bond_order(mol, start_idx, upstream_atom) == 2
    suffix = ("silyl" if mol.atoms[start_idx].symbol == "Si" else "boryl") + ("idene" if is_double else "")
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    if not next_atoms:
        return suffix
    branches = [br for nxt in next_atoms if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx}, start_idx))]
    return f"({format_counted_prefixes(branches)}{suffix})"


def name_halogen_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str:
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    symbol = mol.atoms[start_idx].symbol
    if not next_atoms:
        return HALOGEN_PREFIXES[symbol]
    branches = [br for nxt in next_atoms if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx}, start_idx))]
    valence = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))
    return f"({format_counted_prefixes(branches)}lambda^{valence}-{HALOGEN_LAMBDA_SUFFIXES[symbol]})"


def name_heteroatom_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str | None:
    """Name supported heteroatom-starting recursive fragments."""

    symbol = mol.atoms[start_idx].symbol
    if symbol in HALOGEN_PREFIXES:
        return name_halogen_subgraph(mol, start_idx, exclude_atoms, upstream_atom, branch_namer)
    if symbol == "O":
        return name_oxygen_subgraph(mol, start_idx, exclude_atoms, upstream_atom, branch_namer)
    if symbol == "N":
        return name_nitrogen_subgraph(mol, start_idx, exclude_atoms, upstream_atom, branch_namer)
    if symbol == "S":
        return name_sulfur_subgraph(mol, start_idx, exclude_atoms, upstream_atom, branch_namer)
    if symbol == "Se":
        return name_chalcogen_subgraph(
            mol, start_idx, exclude_atoms, upstream_atom, SIMPLE_SELANYL_PREFIXES, "selanyl", "selenoxo", branch_namer
        )
    if symbol == "Te":
        return name_chalcogen_subgraph(mol, start_idx, exclude_atoms, upstream_atom, set(), "tellanyl", "telluroxo", branch_namer)
    if symbol == "P":
        return name_phosphorus_subgraph(mol, start_idx, exclude_atoms, upstream_atom, branch_namer)
    if symbol in {"Si", "B"}:
        return name_group_13_14_subgraph(mol, start_idx, exclude_atoms, upstream_atom, branch_namer)
    return None
