"""Human-oriented molecule descriptions from structured naming metadata.

This module is separate from :mod:`bluenamer.describer` on purpose.  The
existing describer is useful for debugging the naming pipeline and token
bindings; this descriptor renders a compact, chemistry-facing explanation from
the nested component/substituent tree.  It does not inspect final token spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .engine import DEFAULT_NAMING_ENGINE, NamingRequest, NamingResult
from .graph_io import read_smiles
from .molecule import Molecule


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


def describe_human(smiles: str) -> HumanDescription:
    """Return a readable metadata-backed description for ``smiles``.

    The descriptor intentionally uses ``substituent_tree`` and graph metadata,
    not final ``name_token_spans``.  That keeps the prose independent from the
    currently evolving token-span alignment layer.
    """

    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles=smiles, include_trace=True))
    mol = read_smiles(smiles)
    paragraphs: list[str] = []
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

    substituent_sentence = _substituent_summary_sentence(node)
    if substituent_sentence:
        sentences.append(substituent_sentence)

    child_sentences: list[str] = []
    for child in _iter_child_nodes(node):
        if not _should_expand_child(child):
            continue
        child_subject = _child_subject(child)
        rendered = _describe_node(child, mol, subject=child_subject, depth=depth + 1)
        if rendered:
            child_sentences.append(rendered)

    if child_sentences:
        sentences.extend(child_sentences)
    return " ".join(sentences)


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
        phrases.append(f"{element} at {_positions(locants)}")
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
                bond_phrases.append(f"a {kind} bond between positions {loc_pair[0]} and {loc_pair[1]}")
            elif loc_pair:
                bond_phrases.append(f"a {kind} bond at position {loc_pair[0]}")
    if not bond_phrases:
        return ""
    return "Within that parent framework, there is " + _join_phrases(bond_phrases) + "."


def _unsaturation_position_pairs(
    item: dict[str, Any], parent: dict[str, Any], mol: Molecule
) -> list[tuple[str, ...]]:
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
        return f"The principal characteristic feature is {article}{label} at {_positions(locants)}."
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
            phrases.append(f"a {sign} {_element_name(symbol)} center at position {locant}")
    if not phrases:
        return ""
    return "The parent carries " + _join_phrases(phrases) + "."


def _substituent_summary_sentence(node: dict[str, Any]) -> str:
    children = list(_iter_child_nodes(node, recurse_group_instances=False))
    if not children:
        return ""
    phrases = []
    has_plural = False
    for child in children:
        name = _clean_name(child.get("name") or "substituent")
        locants = [str(loc) for loc in child.get("locants") or ()]
        count = _child_instance_count(child)
        has_plural = has_plural or count > 1
        display_name = _pluralized_substituent_name(name, count)
        if locants:
            phrases.append(f"{display_name} at {_positions(locants)}")
        else:
            phrases.append(display_name)
    verb = "are" if has_plural or len(phrases) > 1 else "is"
    return f"Attached to this framework {verb} " + _join_phrases(phrases) + "."


def _iter_child_nodes(node: dict[str, Any], *, recurse_group_instances: bool = True):
    for child in node.get("substituents") or ():
        if child.get("kind") == "grouped_substituent_instances":
            if recurse_group_instances:
                for instance in child.get("instances") or ():
                    yield instance
            else:
                yield child
            continue
        yield child
    functional_prefix = node.get("functional_prefix")
    if isinstance(functional_prefix, dict):
        for ligand in functional_prefix.get("ligands") or ():
            yield ligand


def _child_subject(child: dict[str, Any]) -> str:
    name = _clean_name(child.get("name") or "substituent")
    locants = [str(loc) for loc in child.get("locants") or ()]
    if locants:
        return f"The {name} substituent at {_positions(locants)}"
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
        return f"a {name} group"
    if name.endswith("yl"):
        return f"{name} groups"
    return f"{name} groups"


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


def _clean_name(name: str) -> str:
    return name.strip().strip("()")


def _positions(locants: list[str]) -> str:
    return "position " + locants[0] if len(locants) == 1 else "positions " + _join_phrases(_sort_locants(locants))


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
