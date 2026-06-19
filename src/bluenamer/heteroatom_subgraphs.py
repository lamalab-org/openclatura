"""Heteroatom-starting recursive substituent naming."""

from collections.abc import Callable

from .formatting import (
    count_names,
    format_counted_prefixes,
    format_element_substituent,
    is_complex_prefix,
    oxy_prefix_from_branch,
    strip_outer_parentheses,
)
from .heteroatom_substituent_specs import central_oxo_substituent_prefix, unsubstituted_prefix
from .heterocumulene_roles import nitrogen_heterocumulene_role
from .hypervalent_roles import HypervalentCenterRole, HypervalentLigandRole, hypervalent_center_role
from .ionic_naming import ammonio_prefix
from .molecule import Molecule
from .namer_config import (
    HALOGEN_LAMBDA_SUFFIXES,
    HALOGEN_PREFIXES,
    SIMPLE_SELANYL_PREFIXES,
    SIMPLE_SULFANYL_PREFIXES,
)
from .nitrogen_roles import terminal_n3_substituent_role
from .nomenclature import RULES
from .oxoacid_roles import OxoLigandRole, central_oxo_substituent_role
from .rules import multipliers

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
        n for n in mol.get_neighbors(start_idx) if n not in exclude_atoms and n != upstream_atom and n not in blocked
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


def _single_imino_n_substituent_name(
    mol: Molecule,
    nitrogen: int,
    exclude_atoms: set[int],
    branch_namer: BranchNamer,
) -> str:
    roots = [
        neighbor
        for neighbor in mol.get_neighbors(nitrogen)
        if neighbor not in exclude_atoms and mol.atoms[neighbor].symbol != "H"
    ]
    if len(roots) != 1:
        return ""
    branch = name_branch_or_none(mol, roots[0], exclude_atoms | {nitrogen}, nitrogen, branch_namer)
    if not branch:
        return ""
    return f"N-{strip_outer_parentheses(branch)}-"


def central_oxo_ligand_atoms(mol: Molecule, atom_idx: int, exclude_atoms: set[int]) -> list[int]:
    """Return graph-classified oxo/oxido ligand atoms for a substituent center."""

    hypervalent = hypervalent_substituent_role_for_center(mol, atom_idx, exclude_atoms)
    if hypervalent is not None and hypervalent.template_audit(mol).ok:
        return [
            ligand.atom
            for ligand in hypervalent.ligands
            if ligand.role in {HypervalentLigandRole.OXO, HypervalentLigandRole.OXIDO}
        ]
    role = central_oxo_substituent_role_for_center(mol, atom_idx, exclude_atoms)
    if role is None:
        return []
    return [ligand.oxygen for ligand in role.ligands if ligand.role in {OxoLigandRole.OXO, OxoLigandRole.OXIDO}]


def central_oxo_substituent_role_for_center(mol: Molecule, atom_idx: int, exclude_atoms: set[int]):
    component_atoms = set(mol.atoms) - set(exclude_atoms)
    component_atoms.add(atom_idx)
    return central_oxo_substituent_role(mol, component_atoms, atom_idx)


def hypervalent_substituent_role_for_center(
    mol: Molecule,
    atom_idx: int,
    exclude_atoms: set[int],
) -> HypervalentCenterRole | None:
    component_atoms = set(mol.atoms) - set(exclude_atoms)
    component_atoms.add(atom_idx)
    return hypervalent_center_role(mol, component_atoms, atom_idx)


def central_oxo_substituent_excluded_ligand_atoms(mol: Molecule, atom_idx: int, exclude_atoms: set[int]) -> list[int]:
    """Return oxygen ligands represented by an exact class prefix."""

    role = central_oxo_substituent_role_for_center(mol, atom_idx, exclude_atoms)
    if role is not None and central_oxo_substituent_prefix(role) is not None:
        return role.oxygen_atoms
    hypervalent = hypervalent_substituent_role_for_center(mol, atom_idx, exclude_atoms)
    if hypervalent is not None and hypervalent.template_audit(mol).ok:
        oxygen_roles = {HypervalentLigandRole.OXO, HypervalentLigandRole.OXIDO}
        return [ligand.atom for ligand in hypervalent.ligands if ligand.role in oxygen_roles]
    if role is None:
        return central_oxo_ligand_atoms(mol, atom_idx, exclude_atoms)
    return role.oxygen_atoms


def central_oxo_substituent_prefix_for_center(mol: Molecule, atom_idx: int, exclude_atoms: set[int]) -> str | None:
    """Return a data-backed oxo-substituent class prefix for a center."""

    role = central_oxo_substituent_role_for_center(mol, atom_idx, exclude_atoms)
    if role is None:
        return None
    return central_oxo_substituent_prefix(role)


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


def format_amino_from_branches(
    branches: list[str], is_double: bool, is_cation: bool = False, is_anion: bool = False
) -> str:
    if is_double:
        if is_cation:
            if len(branches) == 1:
                branch = strip_outer_parentheses(branches[0])
                iminio = charged_heteroatom_prefix("N", 1, "double") or "iminio"
                if is_complex_prefix(branch):
                    return f"(({branch}){iminio})"
                return f"({branch}{iminio})"
            return charged_heteroatom_prefix("N", 1, "double") or "iminio"
        if len(branches) == 1:
            branch = strip_outer_parentheses(branches[0])
            if is_complex_prefix(branch):
                return f"(({branch})imino)"
            return f"({branch}imino)"
        return "imino"

    if is_cation:
        prefix = charged_heteroatom_prefix("N", 1, "single") or "ammonio"
        if prefix == "ammonio":
            return ammonio_prefix([strip_outer_parentheses(branch) for branch in branches])
        return prefix
    if is_anion:
        if not branches:
            return charged_heteroatom_prefix("N", -1, "single") or "azanidyl"
        branch_names = [strip_outer_parentheses(branch) for branch in branches]
        prefix = charged_heteroatom_prefix("N", -1, "single") or "azanidyl"
        return f"({format_counted_prefixes(branch_names)}{prefix})"

    counts = count_names(branches)
    if len(counts) == 1 and list(counts.values())[0] == 1:
        branch = strip_outer_parentheses(branches[0])
        if branch.endswith(("carbonyl", "sulfonyl", "sulfinyl", "carbonothioyl")):
            return f"{branch}amino"
        if is_complex_prefix(branch):
            return f"(({branch})amino)"
        return f"({branch}amino)"

    return f"({format_n_substituted_amino_prefix(branches)}amino)"


def format_n_substituted_amino_prefix(branches: list[str]) -> str:
    """Format multiple substituents attached directly to one amino nitrogen."""

    branch_names = [strip_outer_parentheses(branch) for branch in branches]
    if len(branch_names) > 1 and len(set(branch_names)) == 1 and branch_names[0] == "formyl":
        locants = ",".join("N" for _ in branch_names)
        return f"{locants}-{format_counted_prefixes(branch_names)}"
    return format_counted_prefixes(branches)


def format_lambda_substituent(
    mol: Molecule,
    start_idx: int,
    branches: list[str],
    stereo_prefix_text: str,
    base_suffix: str,
) -> str:
    valence = substituent_bonding_number(mol, start_idx)
    return f"({stereo_prefix_text}{format_counted_prefixes(branches)}-lambda^{valence}-{base_suffix})"


def substituent_bonding_number(mol: Molecule, atom_idx: int) -> int:
    """Return sigma/bond-order valence including explicit hydrogens."""

    bond_order_sum = sum(mol.get_bond(atom_idx, n).order for n in mol.get_neighbors(atom_idx))
    hydrogens = mol.atoms[atom_idx].total_h_count or mol.atoms[atom_idx].explicit_h_count
    if _is_resonance_encoded_sulfur_ylide_center(mol, atom_idx):
        hydrogens = 0
    return bond_order_sum + hydrogens


def _is_resonance_encoded_sulfur_ylide_center(mol: Molecule, atom_idx: int) -> bool:
    """Return true for S(+)=C(-) ylide resonance drawings.

    The parser-facing fallback for these graphs is a lambda-sulfanylidene
    resonance name. The ylide double bond contributes to the lambda state; an
    explicit sulfur hydrogen in the charged drawing is not part of that neutral
    resonance form.
    """

    atom = mol.atoms[atom_idx]
    if atom.symbol != "S" or atom.charge <= 0:
        return False
    return any(
        mol.atoms[neighbor].is_carbon
        and mol.atoms[neighbor].charge < 0
        and (bond := mol.get_bond(atom_idx, neighbor)) is not None
        and bond.order == 2
        for neighbor in mol.get_neighbors(atom_idx)
    )


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
            return charged_heteroatom_prefix("O", -1, "single") or "oxido"
        return unsubstituted_prefix("O") or "hydroxy"

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
        return oxy_prefix_from_branch(branch)
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
    is_cation = mol.atoms[start_idx].charge > 0
    is_anion = mol.atoms[start_idx].charge < 0
    terminal_n3 = terminal_n3_prefix(mol, start_idx, exclude_atoms, upstream_atom)
    if terminal_n3:
        return terminal_n3
    heterocumulene = nitrogen_heterocumulene_role(mol, start_idx, exclude_atoms, upstream_atom)
    if heterocumulene is not None:
        return heterocumulene.prefix
    if not next_atoms:
        if is_double:
            return "imino"
        if is_triple:
            return "nitrilo"
        if is_cation:
            return charged_heteroatom_prefix("N", 1, "single") or "ammonio"
        if is_anion:
            return charged_heteroatom_prefix("N", -1, "single") or "azanidyl"
        return unsubstituted_prefix("N") or "amino"

    branches = []
    for nxt in next_atoms:
        c_oxygens = double_bonded_neighbors(mol, nxt, "O")
        c_sulfurs = double_bonded_neighbors(mol, nxt, "S")
        if mol.atoms[nxt].is_carbon and len(c_oxygens) == 1:
            branches.append(
                name_carbonyl_like_fragment(
                    mol,
                    nxt,
                    start_idx,
                    c_oxygens,
                    exclude_atoms,
                    "carbonyl",
                    "formyl",
                    branch_namer,
                    amino_base="carbamoyl",
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
                sulfur_imide = _sulfur_imide_branch_name(mol, start_idx, nxt, exclude_atoms, s_oxygens, branch_namer)
                if sulfur_imide:
                    branches.append(sulfur_imide)
            else:
                branch = branch_namer(mol, nxt, exclude_atoms | {start_idx}, start_idx)
                if branch:
                    branches.append(branch)

    return format_amino_from_branches(branches, is_double, is_cation, is_anion)


def terminal_n3_prefix(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
) -> str:
    """Render terminal N-N-N charge-separated chains from ordered graph roles."""

    role = terminal_n3_substituent_role(mol, start_idx, exclude_atoms, upstream_atom)
    return role.key if role is not None else ""


def _sulfur_imide_branch_name(
    mol: Molecule,
    nitrogen: int,
    sulfur: int,
    exclude_atoms: set[int],
    s_oxygens: list[int],
    branch_namer: BranchNamer,
) -> str:
    """Render an N=S(=O)n branch without collapsing it to sulfo-amino acid wording."""

    n_s_bond = mol.get_bond(nitrogen, sulfur)
    if n_s_bond is None:
        return ""
    local_exclude = exclude_atoms | {nitrogen, sulfur} | set(s_oxygens)
    next_atoms = subgraph_neighbors(mol, sulfur, exclude_atoms, nitrogen, s_oxygens)
    stereo_prefix_text = stereo_prefix(mol.atoms[sulfur])

    if n_s_bond.order == 2:
        cyclic_name = _cyclic_sulfur_imide_ligand_name(mol, sulfur, nitrogen, next_atoms, s_oxygens)
        if cyclic_name:
            return cyclic_name
        branches = [br for nxt in next_atoms if (br := branch_namer(mol, nxt, local_exclude, sulfur))]
        branches.extend(["oxo"] * len(s_oxygens))
        if not branches:
            return "sulfanylidene"
        if len(branches) == len(s_oxygens):
            return f"{format_counted_prefixes(branches)}sulfanylidene"
        return strip_outer_parentheses(
            format_lambda_substituent(mol, sulfur, branches, stereo_prefix_text, "sulfanylidene")
        )

    branch_idx = first_substituent_neighbor(mol, sulfur, {nitrogen, *s_oxygens})
    branch = name_branch_or_none(mol, branch_idx, local_exclude, sulfur, branch_namer)
    if branch:
        return f"{stereo_prefix_text}{branch}{sulfur_oxo_suffix(len(s_oxygens))}"
    return "sulfo"


def _cyclic_sulfur_imide_ligand_name(
    mol: Molecule,
    sulfur: int,
    nitrogen: int,
    ring_roots: list[int],
    oxo_atoms: list[int],
) -> str:
    """Return a single cyclic S=N ligand name for simple saturated S-rings.

    Hypervalent cyclic imides such as O=S1(CCC1)=N are one sulfur-containing
    ring ligand, not two independent alkyl ligands on sulfur. This conservative
    role only accepts a single saturated carbon path between the two sulfur
    ring neighbors and leaves hetero/unsaturated variants to the general path.
    """

    if len(ring_roots) != 2:
        return ""
    path = _simple_carbon_path_between(mol, ring_roots[0], ring_roots[1], blocked={sulfur, nitrogen, *oxo_atoms})
    if not path:
        return ""
    ring_size = len(path) + 1
    parent = {
        3: "thiirane",
        4: "thietane",
        5: "thiolane",
        6: "thiane",
        7: "thiepane",
        8: "thiocane",
    }.get(ring_size)
    if not parent:
        return ""
    valence = substituent_bonding_number(mol, sulfur)
    prefixes = []
    if oxo_atoms:
        prefixes.append("oxo" if len(oxo_atoms) == 1 else f"{multipliers.basic(len(oxo_atoms))}oxo")
    prefix = "".join(f"1-{item}-" for item in prefixes)
    return f"{prefix}1lambda^{valence}-{parent}-1-ylidene"


def _simple_carbon_path_between(mol: Molecule, start: int, end: int, blocked: set[int]) -> list[int]:
    queue: list[tuple[int, list[int]]] = [(start, [start])]
    seen = {start}
    while queue:
        atom_idx, path = queue.pop(0)
        if atom_idx == end:
            if all(mol.atoms[idx].is_carbon for idx in path) and all(
                mol.get_bond(a, b).order == 1 for a, b in zip(path, path[1:])
            ):
                return path
            return []
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in blocked or neighbor in seen:
                continue
            bond = mol.get_bond(atom_idx, neighbor)
            if bond is None or bond.order != 1 or not mol.atoms[neighbor].is_carbon:
                continue
            seen.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return []


def charged_heteroatom_prefix(symbol: str, charge: int, bond_kind: str) -> str | None:
    sign = "+" if charge > 0 else "-"
    return RULES.charges.heteroatom_charge_prefixes.get(f"{symbol}:{sign}:{bond_kind}")


def name_sulfur_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str:
    is_double = upstream_bond_order(mol, start_idx, upstream_atom) == 2
    s_oxygens = central_oxo_substituent_excluded_ligand_atoms(mol, start_idx, exclude_atoms)
    s_nitrogens = [n for n in double_bonded_neighbors(mol, start_idx, "N") if n != upstream_atom]
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom, s_oxygens + s_nitrogens)
    stereo_prefix_text = stereo_prefix(mol.atoms[start_idx])
    class_prefix = central_oxo_substituent_prefix_for_center(mol, start_idx, exclude_atoms)

    if len(s_oxygens) == 1 and len(s_nitrogens) == 1:
        suffix = "sulfonimidoyl" + ("idene" if is_double else "")
        n_substituent = _single_imino_n_substituent_name(
            mol,
            s_nitrogens[0],
            exclude_atoms | {start_idx} | set(s_oxygens),
            branch_namer,
        )
        if not next_atoms:
            return f"{stereo_prefix_text}{n_substituent}{suffix}"
        if len(next_atoms) == 1:
            branch = name_branch_or_none(
                mol,
                next_atoms[0],
                exclude_atoms | {start_idx} | set(s_oxygens) | set(s_nitrogens),
                start_idx,
                branch_namer,
            )
            if branch and n_substituent:
                return f"({stereo_prefix_text}{n_substituent}S-{strip_outer_parentheses(branch)}{suffix})"
            return f"({stereo_prefix_text}{branch}{suffix})" if branch else f"{stereo_prefix_text}{suffix}"
        branches = [
            br
            for nxt in next_atoms
            if (
                br := branch_namer(mol, nxt, exclude_atoms | {start_idx} | set(s_oxygens) | set(s_nitrogens), start_idx)
            )
        ]
        branches.extend(["oxo"] * len(s_oxygens))
        branches.extend(["imino"] * len(s_nitrogens))
        return format_lambda_substituent(
            mol, start_idx, branches, stereo_prefix_text, "sulfanylidene" if is_double else "sulfanyl"
        )

    if s_oxygens:
        suffix = (class_prefix or sulfur_oxo_suffix(len(s_oxygens))) + ("idene" if is_double else "")
        if not next_atoms:
            return f"{stereo_prefix_text}{suffix}"
        if len(next_atoms) == 1:
            branch = name_branch_or_none(
                mol, next_atoms[0], exclude_atoms | {start_idx} | set(s_oxygens), start_idx, branch_namer
            )
            return f"({stereo_prefix_text}{branch}{suffix})" if branch else f"{stereo_prefix_text}{suffix}"
        branches = [
            br
            for nxt in next_atoms
            if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx} | set(s_oxygens), start_idx))
        ]
        branches.extend(["oxo"] * len(s_oxygens))
        return format_lambda_substituent(
            mol, start_idx, branches, stereo_prefix_text, "sulfanylidene" if is_double else "sulfanyl"
        )

    if s_nitrogens:
        base = "sulfanylidene" if is_double else "sulfanyl"
        if not next_atoms:
            return f"{stereo_prefix_text}imino{base}"
        branches = [
            br
            for nxt in next_atoms
            if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx} | set(s_nitrogens), start_idx))
        ]
        if len(s_nitrogens) == 1:
            branches = ["imino", *branches]
        else:
            branches = [f"{multipliers.basic(len(s_nitrogens))}imino", *branches]
        if not is_double and len(next_atoms) == 1 and len(s_nitrogens) == 1:
            return f"({stereo_prefix_text}{format_counted_prefixes(branches)}{base})"
        return format_lambda_substituent(mol, start_idx, branches, stereo_prefix_text, base)

    if not next_atoms:
        if not is_double and mol.atoms[start_idx].charge > 0:
            return f"{stereo_prefix_text}sulfaniumyl"
        return "thioxo" if is_double else f"{stereo_prefix_text}{unsubstituted_prefix('S') or 'sulfanyl'}"

    if len(next_atoms) == 1:
        branch = branch_namer(mol, next_atoms[0], exclude_atoms | {start_idx}, start_idx)
        if not is_double and mol.atoms[start_idx].charge > 0:
            return f"({stereo_prefix_text}{branch}sulfaniumyl)" if branch else f"{stereo_prefix_text}sulfaniumyl"
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
    oxo_ligands = central_oxo_substituent_excluded_ligand_atoms(mol, start_idx, exclude_atoms)
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom, oxo_ligands)
    stereo_prefix_text = stereo_prefix(mol.atoms[start_idx])
    if oxo_ligands:
        suffix = (
            central_oxo_substituent_prefix_for_center(mol, start_idx, exclude_atoms)
            or unsubstituted_prefix(mol.atoms[start_idx].symbol, len(oxo_ligands))
            or element_suffix
        )
        if not next_atoms:
            return f"{stereo_prefix_text}{suffix}"
        branches = [
            br
            for nxt in next_atoms
            if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx} | set(oxo_ligands), start_idx))
        ]
        branches.extend(["oxo"] * len(oxo_ligands))
        return format_lambda_substituent(
            mol, start_idx, branches, stereo_prefix_text, element_suffix + ("idene" if is_double else "")
        )
    if not next_atoms:
        if not is_double and mol.atoms[start_idx].charge < 0:
            anion_prefix = {"Se": "selenido", "Te": "tellurido"}.get(mol.atoms[start_idx].symbol)
            if anion_prefix:
                return f"{stereo_prefix_text}{anion_prefix}"
        return (
            oxo_prefix
            if is_double
            else f"{stereo_prefix_text}{unsubstituted_prefix(mol.atoms[start_idx].symbol) or element_suffix}"
        )
    if len(next_atoms) == 1:
        branch = branch_namer(mol, next_atoms[0], exclude_atoms | {start_idx}, start_idx)
        if branch and substituent_bonding_number(mol, start_idx) != mol.atoms[start_idx].element.standard_valence:
            return format_lambda_substituent(
                mol, start_idx, [branch], stereo_prefix_text, element_suffix + ("idene" if is_double else "")
            )
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
    upstream_order = upstream_bond_order(mol, start_idx, upstream_atom)
    multiple_bond_suffix = {2: "idene", 3: "idyne"}.get(upstream_order, "")
    p_oxygens = central_oxo_substituent_excluded_ligand_atoms(mol, start_idx, exclude_atoms)
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom, p_oxygens)
    stereo_prefix_text = stereo_prefix(mol.atoms[start_idx])
    suffix = (
        central_oxo_substituent_prefix_for_center(mol, start_idx, exclude_atoms)
        or unsubstituted_prefix("P", len(p_oxygens))
        or ("phosphoryl" if p_oxygens else "phosphanyl")
    ) + multiple_bond_suffix
    if not next_atoms:
        return f"{stereo_prefix_text}{suffix}"
    branches = [
        br
        for nxt in next_atoms
        if (br := branch_namer(mol, nxt, exclude_atoms | {start_idx} | set(p_oxygens), start_idx))
    ]
    return f"({stereo_prefix_text}{format_counted_prefixes(branches)}{suffix})"


def name_group_13_14_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    branch_namer: BranchNamer,
) -> str:
    is_double = upstream_bond_order(mol, start_idx, upstream_atom) == 2
    base_suffix = unsubstituted_prefix(mol.atoms[start_idx].symbol) or (
        "silyl" if mol.atoms[start_idx].symbol == "Si" else "boryl"
    )
    suffix = base_suffix + ("idene" if is_double else "")
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
    symbol = mol.atoms[start_idx].symbol
    next_atoms = subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
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
        return name_chalcogen_subgraph(
            mol, start_idx, exclude_atoms, upstream_atom, set(), "tellanyl", "telluroxo", branch_namer
        )
    if symbol == "P":
        return name_phosphorus_subgraph(mol, start_idx, exclude_atoms, upstream_atom, branch_namer)
    if symbol in {"Si", "B"}:
        return name_group_13_14_subgraph(mol, start_idx, exclude_atoms, upstream_atom, branch_namer)
    return None
