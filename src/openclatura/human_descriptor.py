"""Human-oriented molecule descriptions from structured naming metadata.

This module is separate from :mod:`bluenamer.describer` on purpose.  The
existing describer is useful for debugging the naming pipeline and token
bindings; this descriptor renders a compact, chemistry-facing explanation from
the nested component/substituent tree.  It does not inspect final token spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from .engine import DEFAULT_NAMING_ENGINE, NamingRequest, NamingResult
from .graph_io import read_smiles
from .molecule import Molecule
from .rules import stems


@dataclass(frozen=True)
class HumanDescription:
    """Human-oriented description of the generated name."""

    smiles: str
    name: str
    paragraphs: tuple[str, ...]
    result: NamingResult

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "name": self.name,
            "paragraphs": list(self.paragraphs),
            "text": self.text,
        }


def describe_human(smiles: str, verify_opsin: bool = False) -> HumanDescription:
    """Return a readable metadata-backed description for ``smiles``.

    The descriptor intentionally uses ``substituent_tree`` and graph metadata,
    not final ``name_token_spans``.  That keeps the prose independent from the
    currently evolving token-span alignment layer.
    """

    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles=smiles, include_trace=True, verify_opsin=verify_opsin))
    mol = read_smiles(smiles)
    paragraphs: list[str] = []
    smiles_sentence = _processed_smiles_sentence(smiles)
    if smiles_sentence:
        paragraphs.append(smiles_sentence)
    if verify_opsin and not result.ok:
        paragraphs.append("The OPSIN verification failed for this structure.")
    if result.error:
        paragraphs.append(f"The structure could not be named: {result.error}.")
    elif not result.name:
        paragraphs.append("The structure could not be named by the current ruleset.")
    else:
        paragraphs.append(f"The molecule is named {result.name}.")

    for node in result.substituent_tree:
        description = _describe_node(node, mol, subject="The molecule", depth=0)
        if description:
            paragraphs.append(description)

    return HumanDescription(smiles=smiles, name=result.name, paragraphs=tuple(paragraphs), result=result)


def _processed_smiles_sentence(smiles: str) -> str:
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        return ""
    processed = Chem.MolToSmiles(rdmol, canonical=False)
    atom_order = _smiles_atom_output_order(rdmol)
    annotated = _annotated_smiles_atom_order(processed, atom_order)
    if annotated:
        return f"Processed SMILES: {processed}\nAtom ids in that SMILES: {annotated}"
    return f"Processed SMILES: {processed}\nAtom ids in that SMILES order: {atom_order}"


def _smiles_atom_output_order(rdmol: Chem.Mol) -> list[int]:
    """Return RDKit's atom order for the most recent SMILES write."""

    if not rdmol.HasProp("_smilesAtomOutputOrder"):
        return []
    raw = rdmol.GetProp("_smilesAtomOutputOrder")
    return [int(part) for part in raw.strip("[]").split(",") if part.strip()]


def _annotated_smiles_atom_order(smiles: str, atom_order: list[int]) -> str:
    if not atom_order:
        return ""
    pieces: list[str] = []
    atom_idx = 0
    pos = 0
    while pos < len(smiles):
        token, end = _next_smiles_atom_token(smiles, pos)
        if token is None:
            pieces.append(smiles[pos])
            pos += 1
            continue
        if atom_idx >= len(atom_order):
            return ""
        pieces.append(f"{token}{{{atom_order[atom_idx]}}}")
        atom_idx += 1
        pos = end
    if atom_idx != len(atom_order):
        return ""
    return "".join(pieces)


def _next_smiles_atom_token(smiles: str, pos: int) -> tuple[str | None, int]:
    char = smiles[pos]
    if char == "[":
        end = smiles.find("]", pos + 1)
        if end != -1:
            return smiles[pos : end + 1], end + 1
        return None, pos
    if smiles.startswith(("Cl", "Br"), pos):
        return smiles[pos : pos + 2], pos + 2
    if char in {"B", "C", "N", "O", "P", "S", "F", "I", "b", "c", "n", "o", "p", "s"}:
        return char, pos + 1
    return None, pos


def _describe_node(node: dict[str, Any], mol: Molecule, *, subject: str, depth: int) -> str:
    sentences: list[str] = []
    parent = node.get("parent") or {}
    parent_sentence = _parent_sentence(subject, parent)
    if parent_sentence:
        sentences.append(parent_sentence)

    hetero_sentence = _heteroatom_sentence(parent)
    if hetero_sentence:
        sentences.append(hetero_sentence)

    unsaturation_sentence = _unsaturation_sentence(node, parent, mol)
    if unsaturation_sentence:
        sentences.append(unsaturation_sentence)

    principal_sentence = _principal_group_sentence(node)
    if principal_sentence:
        sentences.append(principal_sentence)

    charge_sentence = _charge_sentence(node)
    if charge_sentence:
        sentences.append(charge_sentence)

    substituent_sentence = _substituent_summary_sentence(node, mol)
    if substituent_sentence:
        sentences.append(substituent_sentence)

    child_sentences: list[str] = []
    for child in _iter_child_nodes(node):
        if not _should_expand_child(child):
            continue
        child_subject = _child_subject(child, parent, mol)
        rendered = _describe_node(child, mol, subject=child_subject, depth=depth + 1)
        if rendered:
            child_sentences.append(rendered)

    if child_sentences:
        sentences.extend(child_sentences)
    return "\n".join(sentences)


def _parent_sentence(subject: str, parent: dict[str, Any]) -> str:
    if not parent:
        return ""
    retained = parent.get("retained_name")
    parent_kind = _parent_kind(parent)
    if retained:
        return f"{subject} is built around the retained {retained} parent, {parent_kind}."
    return f"{subject} is built around a {parent_kind}."


def _parent_kind(parent: dict[str, Any]) -> str:
    length = parent.get("parent_length")
    length_text = f"{length}-membered " if isinstance(length, int) and length > 0 else ""
    hetero = _hetero_locants(parent)
    skeleton = "heteroskeleton" if hetero else "carbon skeleton"
    if parent.get("is_spiro"):
        descriptor = ".".join(str(x) for x in parent.get("spiro_descriptor") or ())
        prefix = f"spiro[{descriptor}] " if descriptor else "spiro "
        return f"{length_text}{prefix}{skeleton}".strip()
    if parent.get("is_bicycle"):
        descriptor = ".".join(str(x) for x in parent.get("bicycle_descriptor") or ())
        prefix = f"bicyclic [{descriptor}] " if descriptor else "bicyclic "
        return f"{length_text}{prefix}{skeleton}".strip()
    if parent.get("is_polycycle"):
        descriptor = parent.get("polycycle_descriptor")
        prefix = f"polycyclic {descriptor} " if descriptor else "polycyclic "
        return f"{length_text}{prefix}{skeleton}".strip()
    if parent.get("is_ring"):
        return f"{length_text}ring {skeleton}".strip()
    return f"{length_text}acyclic {skeleton}".strip()


def _heteroatom_sentence(parent: dict[str, Any]) -> str:
    groups = _hetero_locants(parent)
    if not groups:
        return ""
    phrases = []
    for symbol, locants in groups:
        element = _element_name(symbol)
        phrases.append(f"{element} at {_positions(locants, parent)}")
    return "Within that parent framework, there is " + _join_phrases(phrases) + "."


def _hetero_locants(parent: dict[str, Any]) -> list[tuple[str, list[str]]]:
    symbols = parent.get("atom_symbols_by_locant") or {}
    groups: dict[str, list[str]] = {}
    for locant, symbol in symbols.items():
        if symbol and symbol not in {"C", "H"}:
            groups.setdefault(str(symbol), []).append(str(locant))
    return [(symbol, _sort_locants(locants)) for symbol, locants in sorted(groups.items(), key=lambda item: item[0])]


def _unsaturation_sentence(node: dict[str, Any], parent: dict[str, Any], mol: Molecule) -> str:
    unsaturations = node.get("unsaturations") or []
    if not unsaturations:
        return ""
    bond_phrases: list[str] = []
    for item in unsaturations:
        kind = "triple" if item.get("bond_key") == "triple" else "double"
        for loc_pair in _unsaturation_position_pairs(item, parent, mol):
            if len(loc_pair) == 2:
                bond_phrases.append(
                    f"a {kind} bond between {_position_label(loc_pair[0], parent)} "
                    f"and {_position_label(loc_pair[1], parent)}"
                )
            elif loc_pair:
                bond_phrases.append(f"a {kind} bond at {_position_label(loc_pair[0], parent)}")
    if not bond_phrases:
        return ""
    return "Within that parent framework, there is " + _join_phrases(bond_phrases) + "."


def _unsaturation_position_pairs(item: dict[str, Any], parent: dict[str, Any], mol: Molecule) -> list[tuple[str, ...]]:
    atom_to_locant = {int(atom): str(locant) for locant, atom in (parent.get("atom_ids_by_locant") or {}).items()}
    pairs: list[tuple[str, ...]] = []
    used_bonds: set[int] = set()
    for bond_id in item.get("bonds") or ():
        bond = mol.bonds.get(int(bond_id))
        if bond is None:
            continue
        locants = [atom_to_locant.get(bond.u), atom_to_locant.get(bond.v)]
        if all(locants):
            pairs.append(tuple(_sort_locants([loc for loc in locants if loc])))
            used_bonds.add(int(bond_id))
    if pairs:
        return sorted(pairs, key=lambda pair: [_locant_sort_key(locant) for locant in pair])
    for locant in item.get("locants") or ():
        parsed = _parse_unsaturation_locant(str(locant))
        if parsed:
            pairs.append(parsed)
    return pairs


def _parse_unsaturation_locant(locant: str) -> tuple[str, ...]:
    if "(" in locant and locant.endswith(")"):
        first, second = locant[:-1].split("(", 1)
        if first and second:
            return (first, second)
    return (locant,) if locant else ()


def _principal_group_sentence(node: dict[str, Any]) -> str:
    group = node.get("principal_group")
    if not group:
        return ""
    key = str(group.get("key") or "functional group")
    locants = [str(loc) for loc in group.get("locants") or ()]
    label = _principal_group_label(key, len(locants))
    article = "" if label.endswith("s") else "an " if label[0].lower() in "aeiou" else "a "
    if locants:
        return f"The principal characteristic feature is {article}{label} at {_positions(locants, node.get('parent') or {})}."
    return f"The principal characteristic feature is {article}{label}."


def _principal_group_label(key: str, count: int) -> str:
    labels = {
        "ketone": "oxo group" if count <= 1 else "oxo groups",
        "alcohol": "hydroxy group" if count <= 1 else "hydroxy groups",
        "amine": "amino group" if count <= 1 else "amino groups",
        "imino": "imino group" if count <= 1 else "imino groups",
        "carboxylic_acid": "carboxylic acid group" if count <= 1 else "carboxylic acid groups",
        "amide": "amide group" if count <= 1 else "amide groups",
        "nitrile": "nitrile group" if count <= 1 else "nitrile groups",
        "aldehyde": "formyl group" if count <= 1 else "formyl groups",
    }
    return labels.get(key, _readable_key(key) + (" group" if count <= 1 else " groups"))


def _charge_sentence(node: dict[str, Any]) -> str:
    charges = node.get("parent_charges") or []
    if not charges:
        return ""
    phrases = []
    for charge in charges:
        locant = charge.get("locant")
        symbol = charge.get("symbol") or "atom"
        value = int(charge.get("charge") or 0)
        if value:
            sign = "positive" if value > 0 else "negative"
            phrases.append(
                f"a {sign} {_element_name(symbol)} center at {_position_label(str(locant), node.get('parent') or {})}"
            )
    if not phrases:
        return ""
    return "The parent carries " + _join_phrases(phrases) + "."


def _substituent_summary_sentence(node: dict[str, Any], mol: Molecule) -> str:
    children = list(_iter_child_nodes(node, recurse_group_instances=False))
    if not children:
        return ""
    phrases = []
    has_plural = False
    for child in children:
        name = _display_substituent_name(child, mol)
        locants = [str(loc) for loc in child.get("locants") or ()]
        count = _child_instance_count(child)
        has_plural = has_plural or count > 1
        display_name = _pluralized_substituent_name(name, count)
        if locants:
            phrases.append(f"{display_name} at {_positions(locants, node.get('parent') or {})}")
        else:
            phrases.append(display_name)
    verb = "are" if has_plural or len(phrases) > 1 else "is"
    return f"Attached to this framework {verb} " + _join_phrases(phrases) + "."


def _iter_child_nodes(node: dict[str, Any], *, recurse_group_instances: bool = True):
    for child in node.get("substituents") or ():
        if child.get("kind") == "grouped_substituent_instances":
            if recurse_group_instances:
                yield from child.get("instances") or ()
            else:
                yield child
            continue
        yield child
    functional_prefix = node.get("functional_prefix")
    if isinstance(functional_prefix, dict):
        yield from functional_prefix.get("ligands") or ()


def _child_subject(child: dict[str, Any], containing_parent: dict[str, Any], mol: Molecule) -> str:
    name = _display_substituent_name(child, mol)
    locants = [str(loc) for loc in child.get("locants") or ()]
    if locants:
        return f"The {name} substituent at {_positions(locants, containing_parent)}"
    return f"The {name} substituent"


def _should_expand_child(child: dict[str, Any]) -> bool:
    parent = child.get("parent") or {}
    if child.get("substituents") or child.get("functional_prefix"):
        return True
    if child.get("principal_group") or child.get("unsaturations") or child.get("replacement_prefixes"):
        return True
    if child.get("parent_charges"):
        return True
    if parent.get("retained_name"):
        return True
    if parent.get("is_ring") or parent.get("is_bicycle") or parent.get("is_spiro") or parent.get("is_polycycle"):
        return True
    return False


def _child_instance_count(child: dict[str, Any]) -> int:
    if child.get("kind") == "grouped_substituent_instances":
        instances = child.get("instances") or ()
        return len(instances) if instances else int(child.get("instance_count") or 1)
    return int(child.get("instance_count") or 1)


def _pluralized_substituent_name(name: str, count: int) -> str:
    if count <= 1:
        return f"{_article_for(name)} {name} group"
    return f"{name} groups"


def _article_for(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _display_substituent_name(child: dict[str, Any], mol: Molecule) -> str:
    """Return the local substituent role from tree and graph metadata."""

    if child.get("kind") == "grouped_substituent_instances":
        for instance in child.get("instances") or ():
            if isinstance(instance, dict):
                return _display_substituent_name(instance, mol)

    parent = child.get("parent") if isinstance(child.get("parent"), dict) else {}
    parent_label = _parent_substituent_label(parent)
    if parent_label:
        return parent_label
    graph_label = _graph_substituent_label(child, mol)
    if graph_label:
        return graph_label
    return _readable_key(str(child.get("kind") or "substituent"))


def _parent_substituent_label(parent: dict[str, Any]) -> str:
    suffix = parent.get("substituent_suffix") if isinstance(parent.get("substituent_suffix"), dict) else {}
    suffix_text = str(suffix.get("suffix") or "")
    if not suffix_text:
        return ""
    retained = str(parent.get("retained_name") or "")
    if retained == "benzene" and suffix_text == "yl":
        return "phenyl"
    length = parent.get("parent_length")
    if isinstance(length, int) and length > 0 and not retained:
        try:
            return stems.stem_for(length) + suffix_text
        except KeyError:
            return ""
    if retained:
        return f"{retained}-derived"
    return ""


def _graph_substituent_label(child: dict[str, Any], mol: Molecule) -> str:
    own_atoms = _local_role_atoms(child)
    if not own_atoms:
        return ""
    symbols = [mol.atoms[idx].symbol for idx in sorted(own_atoms) if idx in mol.atoms]
    symbol_set = set(symbols)
    if len(own_atoms) == 1 and symbols:
        symbol = symbols[0]
        has_nested_ligand = bool(list(_iter_child_nodes(child)))
        return {
            "N": "amino",
            "O": "oxy" if has_nested_ligand else "hydroxy",
            "S": "sulfanyl",
            "F": "fluoro",
            "Cl": "chloro",
            "Br": "bromo",
            "I": "iodo",
        }.get(symbol, "")
    if "S" in symbol_set and symbols.count("O") >= 2:
        return "sulfonyl"
    if "S" in symbol_set and symbols.count("O") == 1:
        return "sulfinyl"
    if "C" in symbol_set and "O" in symbol_set and _has_double_bond_between(mol, own_atoms, "C", "O"):
        return "carbonyl"
    return ""


def _local_role_atoms(child: dict[str, Any]) -> set[int]:
    atoms = {int(atom) for atom in child.get("atoms") or ()}
    nested_atoms: set[int] = set()
    for nested in _iter_child_nodes(child):
        nested_atoms.update(int(atom) for atom in nested.get("atoms") or ())
    local = atoms - nested_atoms
    return local or atoms


def _has_double_bond_between(mol: Molecule, atom_ids: set[int], symbol_a: str, symbol_b: str) -> bool:
    for atom_idx in atom_ids:
        atom = mol.atoms.get(atom_idx)
        if atom is None or atom.symbol != symbol_a:
            continue
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in atom_ids:
                continue
            neighbor_atom = mol.atoms.get(neighbor)
            bond = mol.get_bond(atom_idx, neighbor)
            if neighbor_atom is not None and neighbor_atom.symbol == symbol_b and bond is not None and bond.order == 2:
                return True
    return False


def _element_name(symbol: str) -> str:
    names = {
        "B": "boron",
        "C": "carbon",
        "N": "nitrogen",
        "O": "oxygen",
        "P": "phosphorus",
        "S": "sulfur",
        "Se": "selenium",
        "Si": "silicon",
    }
    return names.get(symbol, symbol)


def _readable_key(key: str) -> str:
    return key.replace("_", " ").replace("-", " ")


def _positions(locants: list[str], parent: dict[str, Any] | None = None) -> str:
    sorted_locants = _sort_locants(locants)
    if len(sorted_locants) == 1:
        return _position_label(sorted_locants[0], parent or {})
    return "positions " + _join_phrases([_locant_with_atom_id(locant, parent or {}) for locant in sorted_locants])


def _position_label(locant: str, parent: dict[str, Any] | None = None) -> str:
    return "position " + _locant_with_atom_id(locant, parent or {})


def _locant_with_atom_id(locant: str, parent: dict[str, Any]) -> str:
    atom_id = _atom_id_for_locant(locant, parent)
    return f"{locant} (atom id {atom_id})" if atom_id is not None else locant


def _atom_id_for_locant(locant: str, parent: dict[str, Any]) -> int | None:
    locant_map = parent.get("atom_ids_by_locant") if isinstance(parent, dict) else None
    if not isinstance(locant_map, dict):
        return None
    atom_id = locant_map.get(str(locant))
    try:
        return int(atom_id)
    except (TypeError, ValueError):
        return None


def _join_phrases(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _sort_locants(locants: list[str]) -> list[str]:
    return sorted(locants, key=_locant_sort_key)


def _locant_sort_key(locant: str):
    primary = locant.split("(", 1)[0].rstrip("'")
    try:
        return (0, int(primary), locant)
    except ValueError:
        return (1, primary, locant)
