"""Graph-derived emitted-token metadata for composed substituent prefixes."""

import re
from collections.abc import Callable

from .assembly_parts import NameTokenBinding
from .formatting import strip_outer_parentheses
from .molecule import Molecule
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
    tokens = []
    for token_text in _lexical_tokens(term_text):
        if token_text.lower() in {"cis", "trans"} and len(stereo_atoms) >= 2:
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
                    bond_ids=bond_ids_within(mol, stereo_atoms),
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
                bond_ids=bond_ids_within(mol, atom_ids),
                charge_atom_ids={idx for idx in atom_ids if mol.atoms[idx].charge != 0},
            )
        )
    return tuple(tokens)


def _embedded_absolute_stereo_tokens(mol: Molecule, center: int, term_text: str) -> tuple[NameTokenBinding, ...]:
    """Return graph-bound tokens for directly rendered non-parent R/S descriptors."""

    descriptor = mol.atoms[center].stereo
    if descriptor not in {"R", "S"}:
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
    ligand_roots = _heteroatom_ligand_roots(mol, center, atom_ids, upstream_atom)
    ligand_atoms_by_token = _direct_ligand_tokens(mol, center, ligand_roots, atom_ids)
    for token_text, ligand_atoms in ligand_atoms_by_token:
        if token_text.lower() in lower_term:
            tokens.append(
                NameTokenBinding(
                    text=token_text,
                    atom_ids=set(ligand_atoms),
                    bond_ids=bond_ids_within(mol, set(ligand_atoms) | {center}),
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
                tokens.append(
                    NameTokenBinding(
                        text=token_text,
                        atom_ids=set(ligand_atoms),
                        bond_ids=bond_ids_within(mol, ligand_atoms),
                        charge_atom_ids={idx for idx in ligand_atoms if mol.atoms[idx].charge != 0},
                    )
                )

    charge_atoms = {center} if mol.atoms[center].charge != 0 else set()
    for token_text in center_tokens:
        tokens.append(
            NameTokenBinding(
                text=token_text,
                atom_ids={center},
                charge_atom_ids=set(charge_atoms),
            )
        )
    return tuple(tokens)


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


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\^[0-9]+)?|\d+(?:,\d+)*(?:'\")?|[0-9]+(?:\([0-9,]+\))?")
