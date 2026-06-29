"""Minimal command-line interface for bluenamer.

Subcommands::

    bluenamer name    "CCO"                # print one name
    bluenamer name    "CCO" --json         # JSON with trace + rules
    bluenamer batch   smiles.txt           # one SMILES per line, write JSONL
    bluenamer version

The CLI is intentionally thin — for anything beyond quick experiments
the Python API (``bluenamer.name`` / ``bluenamer.name_many``) is the
recommended entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from . import __version__, name_many
from . import describe as describe_one
from . import name as name_one


def _iter_smiles(path: str) -> Iterator[str]:
    if path == "-":
        source: Iterable[str] = sys.stdin
    else:
        source = Path(path).read_text(encoding="utf-8").splitlines()
    for line in source:
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def _cmd_name(args: argparse.Namespace) -> int:
    result = name_one(
        args.smiles,
        include_trace=args.json or args.trace,
        verify_opsin=args.verify,
        token_debug=args.token_debug,
    )
    if args.json:
        json.dump(result.to_dict(include_trace=True), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        if result.error:
            print(f"error: {result.error}", file=sys.stderr)
            return 1
        print(result.name)
        if args.trace:
            for hint in result.rule_hints:
                print(f"  - {hint}", file=sys.stderr)
        if result.opsin_check is not None:
            print(f"  opsin: {result.opsin_check.status}", file=sys.stderr)
    return 0 if result.ok else 1


def _cmd_batch(args: argparse.Namespace) -> int:
    smiles_list = list(_iter_smiles(args.input))
    processes = args.processes if args.processes is not None else 1
    results = name_many(
        smiles_list,
        include_trace=args.trace or args.json,
        verify_opsin=args.verify,
        token_debug=args.token_debug,
        processes=processes,
        chunksize=args.chunksize,
    )
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")  # noqa: SIM115
    try:
        for r in results:
            json.dump(r.to_dict(include_trace=args.trace), out, default=str)
            out.write("\n")
    finally:
        if out is not sys.stdout:
            out.close()
    failed = sum(1 for r in results if not r.ok)
    if failed:
        print(f"{failed}/{len(results)} failed", file=sys.stderr)
        return 0 if args.allow_failures else 2
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    description = describe_one(args.smiles, token_debug=args.token_debug)
    if args.json:
        json.dump(description.to_dict(token_debug=args.token_debug), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(str(description) + "\n")
    return 0 if description.name else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bluenamer", description="SMILES → IUPAC name generator")
    parser.add_argument("--version", action="version", version=f"bluenamer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_name = sub.add_parser("name", help="Name a single SMILES")
    p_name.add_argument("smiles")
    p_name.add_argument("--trace", action="store_true", help="also print rule hints to stderr")
    p_name.add_argument("--json", action="store_true", help="emit JSON to stdout")
    p_name.add_argument("--token-debug", action="store_true", help="include verbose emitted token metadata in JSON traces")
    p_name.add_argument("--verify",action=argparse.BooleanOptionalAction,default=True,help="round-trip via OPSIN or --no-verify to skip")
    p_name.set_defaults(func=_cmd_name)

    p_batch = sub.add_parser("batch", help="Name a file of SMILES (JSONL output)")
    p_batch.add_argument("input", help="path to one-SMILES-per-line file, or '-' for stdin")
    p_batch.add_argument("--output", default="-", help="output file path, or '-' for stdout")
    p_batch.add_argument("--trace", action="store_true", help="include trace_segments in JSON output")
    p_batch.add_argument("--json", action="store_true", help="alias of --trace; emit full JSON")
    p_batch.add_argument("--token-debug", action="store_true", help="include verbose emitted token metadata in JSON traces")
    p_batch.add_argument("--verify",action=argparse.BooleanOptionalAction,default=True,help="round-trip via OPSIN or --no-verify to skip")
    p_batch.add_argument(
        "--processes",
        type=lambda v: None if v in {"", "auto"} else int(v),
        default=1,
        help="worker processes (1=serial, 'auto'=all CPUs, integer for fixed count)",
    )
    p_batch.add_argument("--chunksize", type=int, default=64)
    p_batch.add_argument(
        "--allow-failures",
        action="store_true",
        help="exit 0 even if some rows produced an error",
    )
    p_batch.set_defaults(func=_cmd_batch)

    p_describe = sub.add_parser(
        "describe",
        help="Natural-language description of how the name is built",
    )
    p_describe.add_argument("smiles")
    p_describe.add_argument("--json", action="store_true", help="emit structured Description as JSON")
    p_describe.add_argument(
        "--token-debug",
        "--debug-tokens",
        dest="token_debug",
        action="store_true",
        help="include experimental token binding details",
    )
    p_describe.set_defaults(func=_cmd_describe)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Avoid noisy multiprocessing semaphore tracker warnings on macOS.
    os.environ.setdefault("PYTHONWARNINGS", "ignore::ResourceWarning")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
