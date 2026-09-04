"""Compile and validate the reviewable fusion-component vocabulary offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from openclatura.fusion.registry import FusionComponentRegistry

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "src/openclatura/data/fusion_components.json"
TEMPLATE_PATH = ROOT / "src/openclatura/data/retained_fused_graph_templates.json"
HW_PATH = ROOT / "src/openclatura/hantzsch_widman.py"
PARSER_PATH = ROOT / "tests/data/parser_xml_resources/fusionComponents.json"
DEFAULT_OUTPUT = ROOT / "src/openclatura/data/generated_fusion_components.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser_tokens(value: object) -> frozenset[str]:
    tokens: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if node.get("tag") == "token" and isinstance(node.get("text"), str):
                tokens.add(node["text"])
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return frozenset(tokens)


def compiled_registry() -> dict[str, Any]:
    """Return deterministic normalized policy plus offline vocabulary evidence."""

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    parser_data = json.loads(PARSER_PATH.read_text(encoding="utf-8"))
    parser_tokens = _parser_tokens(parser_data)
    registry = FusionComponentRegistry.from_data(policy)
    components = []
    for registered in sorted(registry.components, key=lambda item: item.spec.parent_name):
        spec = registered.spec
        parser_forms = (spec.preferred_fusion_prefix, *spec.accepted_general_prefixes)
        components.append(
            {
                "parent_name": spec.parent_name,
                "preferred_fusion_prefix": spec.preferred_fusion_prefix,
                "accepted_general_prefixes": list(spec.accepted_general_prefixes),
                "template_names": list(registered.template_names),
                "roles": [
                    role
                    for role, enabled in (
                        ("parent", spec.usable_as_parent),
                        ("attached", spec.usable_as_attached),
                    )
                    if enabled
                ],
                "rule": spec.rule_reference,
                "parser_visible_forms": sorted(form for form in parser_forms if form in parser_tokens),
            }
        )
    return {
        "schema_version": 1,
        "registry_version": registry.version,
        "generated_by": "scripts/build_fusion_component_registry.py",
        "runtime_dependency": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (POLICY_PATH, TEMPLATE_PATH, HW_PATH, PARSER_PATH)
        },
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(compiled_registry(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{args.output} is stale; regenerate it with {Path(__file__).name}")
        return
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
