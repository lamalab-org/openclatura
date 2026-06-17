"""Graph-derived emitted-token metadata for composed substituent prefixes."""

import re
from collections.abc import Callable

from .assembly_parts import NameTokenBinding
from .formatting import strip_outer_parentheses
from .molecule import Molecule
from .stereo_descriptors import ABSOLUTE_STEREO_DESCRIPTORS, RELATIVE_STEREO_DESCRIPTORS
from .trace_helpers import bond_ids_within

BranchNamer = Callable[..., str]


def graph_bound_substituent_tokens(
    mol: Molecule,
    root: int,
    atom_ids: set[int],
    term: str,
    upstream_atom: int,
    exclude_atoms: set[int],
    branch_namer: BranchNamer,
) -> tuple[NameTokenBinding, ...]:
    """Return precise token bindings for graph-composed substituent terms."""

    if mol.atoms[root].symbol == "N":
        return _nitrogen_substituent_tokens(mol, root, atom_ids, term, upstream_atom, exclude_atoms, branch_namer)
    if mol.atoms[root].is_carbon:
        return _carbon_substituent_tokens(mol, atom_ids, term)
    return _generic_heteroatom_substituent_tokens(mol, root, atom_ids, term, upstream_atom, exclude_atoms, branch_namer)


def _carbon_substituent_tokens(mol: Molecule, atom_ids: set[int], term: str) -> tuple[NameTokenBinding, ...]:
    """Return graph-bound tokens for carbon-root substituent renderings."""

    term_text = strip_outer_parentheses(term)
    if not term_text:
        return ()
    stereo_atoms = {idx for idx in atom_ids if mol.atoms[idx].raw_stereo and not mol.atoms[idx].stereo}
    substituent_bond_ids = bond_ids_within(mol, atom_ids)
    substituent_charge_atom_ids = {idx for idx in atom_ids if mol.atoms[idx].charge != 0}
    stereo_bond_ids = bond_ids_within(mol, stereo_atoms) if len(stereo_atoms) >= 2 else set()
    tokens = []
    for token_text in _lexical_tokens(term_text):
        if token_text.lower() in RELATIVE_STEREO_DESCRIPTORS and len(stereo_atoms) >= 2:
            tokens.append(
                NameTokenBinding(
                    text=token_text,
                    token_kind="stereo",
                    ownership="exact",
                    confidence="exact",
                    source="renderer_stereo",
                    grammar_role="relative_stereo",
                    binding_key="prefix:relative_stereo",
                    atom_ids=set(stereo_atoms),
                    bond_ids=set(stereo_bond_ids),
                )
            )
            continue
        tokens.append(
            NameTokenBinding(
                text=token_text,
                token_kind="prefix",
                source="substituent_renderer",
                grammar_role="carbon_substituent",
                binding_key="prefix:carbon_substituent",
                atom_ids=set(atom_ids),
                bond_ids=set(substituent_bond_ids),
                charge_atom_ids=set(substituent_charge_atom_ids),
            )
        )
    return tuple(tokens)


def _embedded_absolute_stereo_tokens(mol: Molecule, center: int, term_text: str) -> tuple[NameTokenBinding, ...]:
    """Return graph-bound tokens for directly rendered non-parent R/S descriptors."""

    descriptor = mol.atoms[center].stereo
    if descriptor not in ABSOLUTE_STEREO_DESCRIPTORS:
        return ()
    if not re.search(rf"\({re.escape(descriptor)}\)-", term_text):
        return ()
    return (
        NameTokenBinding(
            text=descriptor,
            token_kind="stereo",
            ownership="exact",
            confidence="exact",
            source="renderer_stereo",
            grammar_role="absolute_stereo",
            binding_key="prefix:absolute_stereo",
            atom_ids={center},
        ),
    )


def _generic_heteroatom_substituent_tokens(
    mol: Molecule,
    center: int,
    atom_ids: set[int],
    term: str,
    upstream_atom: int,
    exclude_atoms: set[int],
    branch_namer: BranchNamer,
) -> tuple[NameTokenBinding, ...]:
    """Return graph-bound tokens for non-nitrogen heteroatom-center prefixes."""

    term_text = strip_outer_parentheses(term)
    lower_term = term_text.lower()
    center_tokens = _heteroatom_center_tokens(mol.atoms[center].symbol, lower_term)
    if not center_tokens:
        return ()

    tokens: list[NameTokenBinding] = list(_embedded_absolute_stereo_tokens(mol, center, term_text))
    carbonyl_like_tokens = _oxygen_carbonyl_like_tokens(mol, center, atom_ids, term_text, upstream_atom)
    if carbonyl_like_tokens:
        return tuple([*tokens, *carbonyl_like_tokens])
    ligand_roots = _heteroatom_ligand_roots(mol, center, atom_ids, upstream_atom)
    ligand_atoms_by_token = _direct_ligand_tokens(mol, center, ligand_roots, atom_ids)
    for token_text, ligand_atoms in ligand_atoms_by_token:
        if token_text.lower() in lower_term:
            ligand_bonds = bond_ids_within(mol, set(ligand_atoms) | {center})
            tokens.append(
                NameTokenBinding(
                    text=token_text,
                    token_kind="prefix",
                    source="substituent_renderer",
                    grammar_role="heteroatom_direct_ligand",
                    binding_key="prefix:heteroatom_direct_ligand",
                    atom_ids=set(ligand_atoms),
                    bond_ids=ligand_bonds,
                    charge_atom_ids={idx for idx in ligand_atoms if mol.atoms[idx].charge != 0},
                )
            )

    branch_roots = [
        root for root in ligand_roots if root not in {atom for _token, atoms in ligand_atoms_by_token for atom in atoms}
    ]
    for ligand_root in branch_roots:
        ligand_atoms = _component_within(mol, ligand_root, atom_ids - {center})
        if not ligand_atoms:
            continue
        ligand_text = branch_namer(mol, ligand_root, exclude_atoms | {center}, upstream_atom=center)
        ligand_text = strip_outer_parentheses(ligand_text or "")
        if not ligand_text:
            continue
        for token_text in _lexical_tokens(ligand_text):
            if token_text.lower() in lower_term:
                token_atoms, token_bonds = _organic_ligand_token_scope(mol, token_text, ligand_atoms, lower_term)
                tokens.append(
                    NameTokenBinding(
                        text=token_text,
                        token_kind="locant" if _is_locant_like_token(token_text) else "prefix",
                        source="substituent_renderer",
                        grammar_role="heteroatom_organic_ligand",
                        binding_key="prefix:heteroatom_organic_ligand",
                        atom_ids=set(token_atoms),
                        bond_ids=set(token_bonds),
                        charge_atom_ids={idx for idx in token_atoms if mol.atoms[idx].charge != 0},
                        locants=(token_text,) if _is_locant_like_token(token_text) else (),
                    )
                )

    charge_atoms = {center} if mol.atoms[center].charge != 0 else set()
    direct_ligand_atoms = {atom for token_text, atoms in ligand_atoms_by_token for atom in atoms if token_text in {"oxo", "oxido", "imino"}}
    center_atoms = {center, *direct_ligand_atoms}
    center_bonds = bond_ids_within(mol, center_atoms)
    for token_text in center_tokens:
        tokens.append(
            NameTokenBinding(
                text=token_text,
                token_kind="prefix",
                source="substituent_renderer",
                grammar_role="heteroatom_center",
                binding_key="prefix:heteroatom_center",
                atom_ids=set(center_atoms),
                bond_ids=set(center_bonds),
                charge_atom_ids=set(charge_atoms),
            )
        )
    return tuple(tokens)


def _organic_ligand_token_scope(
    mol: Molecule,
    token_text: str,
    ligand_atoms: set[int],
    lower_term: str,
) -> tuple[set[int], set[int]]:
    """Return the most local graph scope for a token inside an organic ligand."""

    token_lower = token_text.lower()
    if _is_locant_like_token(token_text):
        locant_atoms = _terminal_substituent_atoms_for_locant_token(mol, ligand_atoms, token_text, lower_term)
        if locant_atoms:
            return locant_atoms, bond_ids_within(mol, locant_atoms | ligand_atoms)
    if token_lower in {"amino", "hydroxy", "fluoro", "chloro", "bromo", "iodo"}:
        atoms = _terminal_atoms_for_prefix(mol, ligand_atoms, token_lower)
        if atoms:
            return atoms, bond_ids_within(mol, atoms | ligand_atoms)
    return set(ligand_atoms), bond_ids_within(mol, ligand_atoms)


def _terminal_substituent_atoms_for_locant_token(
    mol: Molecule,
    ligand_atoms: set[int],
    token_text: str,
    lower_term: str,
) -> set[int]:
    token_lower = token_text.lower()
    candidates = []
    for match in re.finditer(re.escape(token_lower), lower_term):
        after = lower_term[match.end() :]
        if not after.startswith("-"):
            continue
        next_token = _TOKEN_RE.search(after, 1)
        if next_token is None:
            continue
        candidates.extend(_terminal_atoms_for_prefix(mol, ligand_atoms, next_token.group(0).lower()))
    return set(candidates)


def _terminal_atoms_for_prefix(mol: Molecule, ligand_atoms: set[int], token_lower: str) -> set[int]:
    prefix_symbols = {
        "amino": {"N"},
        "hydroxy": {"O"},
        "fluoro": {"F"},
        "chloro": {"Cl"},
        "bromo": {"Br"},
        "iodo": {"I"},
    }.get(token_lower)
    if not prefix_symbols:
        return set()
    return {
        atom_idx
        for atom_idx in ligand_atoms
        if mol.atoms[atom_idx].symbol in prefix_symbols
        and sum(1 for neighbor in mol.get_neighbors(atom_idx) if neighbor in ligand_atoms) == 1
    }


def _oxygen_carbonyl_like_tokens(
    mol: Molecule,
    oxygen: int,
    atom_ids: set[int],
    term_text: str,
    upstream_atom: int,
) -> tuple[NameTokenBinding, ...]:
    """Return graph scopes for O-C(=O)-R prefixes such as carbonyloxy."""

    if mol.atoms[oxygen].symbol != "O":
        return ()
    lower = term_text.lower()
    if "carbonyloxy" not in lower and "carbonothioyloxy" not in lower:
        return ()
    carbonyl = next(
        (
            neighbor
            for neighbor in mol.get_neighbors(oxygen)
            if neighbor != upstream_atom and neighbor in atom_ids and mol.atoms[neighbor].is_carbon
        ),
        None,
    )
    if carbonyl is None:
        return ()
    terminal_chalcogens = [
        neighbor
        for neighbor in mol.get_neighbors(carbonyl)
        if neighbor in atom_ids
        and neighbor != oxygen
        and mol.atoms[neighbor].symbol in {"O", "S"}
        and (bond := mol.get_bond(carbonyl, neighbor)) is not None
        and bond.order == 2
    ]
    if len(terminal_chalcogens) != 1:
        return ()
    carbonyl_atoms = {carbonyl, terminal_chalcogens[0]}
    carbonyl_bonds = bond_ids_within(mol, carbonyl_atoms | {oxygen})
    tokens: list[NameTokenBinding] = [
        NameTokenBinding(
            text="carbon",
            token_kind="prefix",
            source="substituent_renderer",
            grammar_role="carbonyl_like_core",
            binding_key="prefix:carbonyl_like_core",
            atom_ids=set(carbonyl_atoms),
            bond_ids=set(carbonyl_bonds),
        ),
        NameTokenBinding(
            text="carbonyl",
            token_kind="prefix",
            source="substituent_renderer",
            grammar_role="carbonyl_like_core",
            binding_key="prefix:carbonyl_like_core",
            atom_ids=set(carbonyl_atoms),
            bond_ids=set(carbonyl_bonds),
        ),
        NameTokenBinding(
            text="oxy",
            token_kind="prefix",
            source="substituent_renderer",
            grammar_role="heteroatom_center",
            binding_key="prefix:heteroatom_center",
            atom_ids={oxygen},
            bond_ids=bond_ids_within(mol, {oxygen, carbonyl}),
        ),
    ]
    ligand_roots = [
        neighbor
        for neighbor in mol.get_neighbors(carbonyl)
        if neighbor in atom_ids and neighbor not in {oxygen, terminal_chalcogens[0]}
    ]
    if not ligand_roots:
        return tuple(tokens)
    ligand_root = ligand_roots[0]
    ligand_atoms = _component_within(mol, ligand_root, atom_ids - {oxygen, carbonyl, terminal_chalcogens[0]})
    bridge_atoms, branch_atoms, bridge_bonds = _alkenyl_bridge_scope(mol, ligand_root, ligand_atoms)
    if bridge_atoms:
        tokens.extend(_alkenyl_bridge_tokens(mol, bridge_atoms, bridge_bonds, lower))
    if branch_atoms:
        tokens.extend(_branch_phrase_tokens(mol, branch_atoms, lower))
    return tuple(tokens)


def _alkenyl_bridge_scope(
    mol: Molecule,
    ligand_root: int,
    ligand_atoms: set[int],
) -> tuple[set[int], set[int], set[int]]:
    bridge_atoms = {ligand_root}
    branch_root = None
    for neighbor in mol.get_neighbors(ligand_root):
        if neighbor not in ligand_atoms:
            continue
        bond = mol.get_bond(ligand_root, neighbor)
        if bond is not None and bond.order == 2:
            bridge_atoms.add(neighbor)
            branch_candidates = [n for n in mol.get_neighbors(neighbor) if n in ligand_atoms and n != ligand_root]
            branch_root = branch_candidates[0] if branch_candidates else None
            break
    if branch_root is None:
        return bridge_atoms, ligand_atoms - bridge_atoms, bond_ids_within(mol, bridge_atoms)
    branch_atoms = _component_within(mol, branch_root, ligand_atoms - bridge_atoms)
    return bridge_atoms, branch_atoms, bond_ids_within(mol, bridge_atoms)


def _alkenyl_bridge_tokens(
    mol: Molecule,
    bridge_atoms: set[int],
    bridge_bonds: set[int],
    lower_term: str,
) -> list[NameTokenBinding]:
    tokens: list[NameTokenBinding] = []
    if "eth" in lower_term:
        tokens.append(
            NameTokenBinding(
                text="eth",
                token_kind="prefix",
                source="substituent_renderer",
                grammar_role="alkenyl_bridge",
                binding_key="prefix:alkenyl_bridge",
                atom_ids=set(bridge_atoms),
                bond_ids=set(bridge_bonds),
            )
        )
    if "en" in lower_term:
        tokens.append(
            NameTokenBinding(
                text="en",
                token_kind="unsaturation",
                source="substituent_renderer",
                grammar_role="alkenyl_bridge_unsaturation",
                binding_key="prefix:alkenyl_bridge",
                atom_ids=set(bridge_atoms),
                bond_ids=set(bridge_bonds),
            )
        )
    if "yl" in lower_term:
        tokens.append(
            NameTokenBinding(
                text="yl",
                token_kind="prefix",
                source="substituent_renderer",
                grammar_role="alkenyl_bridge_suffix",
                binding_key="prefix:alkenyl_bridge",
                atom_ids=set(bridge_atoms),
                bond_ids=set(bridge_bonds),
            )
        )
    stereo_bonds = {
        bond.idx
        for bond in mol.bonds.values()
        if bond.idx in bridge_bonds and bond.stereo in {"E", "Z"}
    }
    if stereo_bonds:
        tokens.append(
            NameTokenBinding(
                text="1",
                token_kind="locant",
                source="renderer_stereo",
                grammar_role="bond_stereo",
                binding_key="prefix:alkenyl_bridge_stereo",
                atom_ids=set(bridge_atoms),
                bond_ids=set(stereo_bonds),
                locants=("1",),
            )
        )
        descriptor = next(bond.stereo for bond in mol.bonds.values() if bond.idx in stereo_bonds and bond.stereo)
        tokens.append(
            NameTokenBinding(
                text=descriptor,
                token_kind="stereo",
                source="renderer_stereo",
                grammar_role="bond_stereo",
                binding_key="prefix:alkenyl_bridge_stereo",
                atom_ids=set(bridge_atoms),
                bond_ids=set(stereo_bonds),
                locants=("1",),
            )
        )
    return tokens


def _branch_phrase_tokens(mol: Molecule, branch_atoms: set[int], lower_term: str) -> list[NameTokenBinding]:
    tokens: list[NameTokenBinding] = []
    branch_bonds = bond_ids_within(mol, branch_atoms)
    hydroxy_atoms = {
        atom_idx
        for atom_idx in branch_atoms
        if mol.atoms[atom_idx].symbol == "O" and any(neighbor in branch_atoms for neighbor in mol.get_neighbors(atom_idx))
    }
    if "dihydroxyphenyl" in lower_term:
        tokens.append(
            NameTokenBinding(
                text="dihydroxyphenyl",
                token_kind="prefix",
                source="substituent_renderer",
                grammar_role="aryl_branch",
                binding_key="prefix:aryl_branch",
                atom_ids=set(branch_atoms),
                bond_ids=set(branch_bonds),
            )
        )
    if "3,4" in lower_term and hydroxy_atoms:
        hydroxy_bonds = {
            bond.idx
            for oxygen in hydroxy_atoms
            for neighbor in mol.get_neighbors(oxygen)
            if neighbor in branch_atoms and (bond := mol.get_bond(oxygen, neighbor)) is not None
        }
        tokens.append(
            NameTokenBinding(
                text="3,4",
                token_kind="locant",
                source="substituent_renderer",
                grammar_role="aryl_branch_substituent_locants",
                binding_key="prefix:aryl_branch_locants",
                atom_ids=set(hydroxy_atoms),
                bond_ids=hydroxy_bonds,
                locants=("3", "4"),
            )
        )
    if "2-" in lower_term and branch_atoms:
        tokens.append(
            NameTokenBinding(
                text="2",
                token_kind="locant",
                source="substituent_renderer",
                grammar_role="alkenyl_branch_locant",
                binding_key="prefix:alkenyl_branch_locant",
                atom_ids=set(branch_atoms),
                bond_ids=set(branch_bonds),
                locants=("2",),
            )
        )
    return tokens


def _nitrogen_substituent_tokens(
    mol: Molecule,
    nitrogen: int,
    atom_ids: set[int],
    term: str,
    upstream_atom: int,
    exclude_atoms: set[int],
    branch_namer: BranchNamer,
) -> tuple[NameTokenBinding, ...]:
    term_text = strip_outer_parentheses(term)
    lower_term = term_text.lower()
    central_tokens = _nitrogen_central_tokens(lower_term)
    if not central_tokens:
        return ()

    tokens: list[NameTokenBinding] = list(_embedded_absolute_stereo_tokens(mol, nitrogen, term_text))
    uses_n_ligand_locants = "n-" in lower_term or lower_term.startswith("n")

    for ligand_root in _heteroatom_ligand_roots(mol, nitrogen, atom_ids, upstream_atom):
        ligand_atoms = _component_within(mol, ligand_root, atom_ids - {nitrogen})
        if not ligand_atoms:
            continue
        ligand_text = branch_namer(mol, ligand_root, exclude_atoms | {nitrogen}, upstream_atom=nitrogen)
        ligand_text = strip_outer_parentheses(ligand_text or "")
        if not ligand_text:
            continue
        if uses_n_ligand_locants:
            tokens.append(NameTokenBinding(text="N", atom_ids={nitrogen}))
        for token_text in _lexical_tokens(ligand_text):
            if token_text.lower() in lower_term:
                tokens.append(
                    NameTokenBinding(
                        text=token_text,
                        atom_ids=set(ligand_atoms),
                        bond_ids=bond_ids_within(mol, ligand_atoms),
                        charge_atom_ids={idx for idx in ligand_atoms if mol.atoms[idx].charge != 0},
                    )
                )

    charge_atoms = {nitrogen} if mol.atoms[nitrogen].charge != 0 else set()
    for token_text in central_tokens:
        tokens.append(
            NameTokenBinding(
                text=token_text,
                atom_ids={nitrogen},
                charge_atom_ids=set(charge_atoms),
            )
        )
    return tuple(tokens)


def _nitrogen_central_tokens(term: str) -> tuple[str, ...]:
    tokens = []
    for token in ("ammonio", "azanidyl", "iminio", "amino", "imino", "nitrilo"):
        if token in term:
            tokens.append(token)
    return tuple(tokens)


def _heteroatom_center_tokens(symbol: str, term: str) -> tuple[str, ...]:
    candidates = {
        "O": ("hydroperoxy", "peroxy", "hydroxy", "oxy", "oxido"),
        "S": (
            "sulfonimidoyl",
            "sulfanylidene",
            "sulfaniumyl",
            "sulfonyl",
            "sulfinyl",
            "sulfanyl",
            "sulfo",
            "thioxo",
        ),
        "Se": ("selenanylidene", "selanylidene", "selenanyl", "selanyl", "selenido", "seleno", "selanyl"),
        "Te": ("telluranylidene", "tellanyl", "tellurido", "telluro"),
        "P": ("phosphoryl", "phosphono", "phosphanylidene", "phosphanylidynemethyl", "phosphanyl", "phosphino"),
        "B": ("borylidene", "boryl", "boranuide"),
        "Si": ("silylidene", "silyl"),
        "F": ("fluoro",),
        "Cl": ("chloro", "chloranyl", "chlorosyl"),
        "Br": ("bromo", "bromanyl", "bromosyl"),
        "I": ("iodo", "iodanyl", "iodosyl"),
    }.get(symbol, ())
    return tuple(token for token in candidates if token in term)


def _direct_ligand_tokens(
    mol: Molecule,
    center: int,
    ligand_roots: list[int],
    atom_ids: set[int],
) -> list[tuple[str, set[int]]]:
    tokens = []
    for root in ligand_roots:
        atom = mol.atoms[root]
        bond = mol.get_bond(center, root)
        if atom.symbol == "O":
            if bond and bond.order == 2:
                tokens.append(("oxo", {root}))
            elif atom.charge < 0:
                tokens.append(("oxido", {root}))
            else:
                oxygen_neighbors = [n for n in mol.get_neighbors(root) if n in atom_ids and n != center]
                if oxygen_neighbors and mol.atoms[oxygen_neighbors[0]].symbol == "O":
                    tokens.append(("peroxy", {root, oxygen_neighbors[0]}))
                elif oxygen_neighbors:
                    tokens.append(
                        ("oxy", {root, *(_component_within(mol, oxygen_neighbors[0], atom_ids - {center, root}))})
                    )
                else:
                    tokens.append(("hydroxy", {root}))
        elif atom.symbol == "N" and bond and bond.order == 2:
            tokens.append(("imino", {root}))
        elif atom.symbol in {"F", "Cl", "Br", "I"}:
            halo = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}[atom.symbol]
            tokens.append((halo, {root}))
    return tokens


def _heteroatom_ligand_roots(mol: Molecule, center: int, atom_ids: set[int], upstream_atom: int) -> list[int]:
    return [
        neighbor
        for neighbor in mol.get_neighbors(center)
        if neighbor != upstream_atom and neighbor in atom_ids and mol.atoms[neighbor].symbol != "H"
    ]


def _component_within(mol: Molecule, root: int, allowed_atoms: set[int]) -> set[int]:
    if root not in allowed_atoms:
        return set()
    seen = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        for neighbor in mol.get_neighbors(current):
            if neighbor in allowed_atoms and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _lexical_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _TOKEN_RE.finditer(text))


def _is_locant_like_token(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:,\d+)*", text))


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\^[0-9]+)?|\d+(?:,\d+)*(?:'\")?|[0-9]+(?:\([0-9,]+\))?")
