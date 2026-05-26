# bluenamer

A deterministic SMILES → IUPAC name generator implementing the rules of the
IUPAC Blue Book (2013 recommendations).

The package walks the molecular graph (parsed via RDKit), perceives functional
groups and ring systems, selects a principal parent, assigns locants, and
assembles the substitutive name. Every step is recorded in an inspectable
decision trace so the *why* of a name is recoverable, not just the *what*.

> **Status:** alpha. The naming engine handles a broad slice of organic
> structures (alkanes/alkenes/alkynes, common functional groups, simple
> heterocycles, fused/spiro/bridged systems, retained names from the Blue
> Book). PubChem/QM9 coverage is being measured; see `examples/`.

## Install

```bash
pip install bluenamer
```

Optional extras:

| extra        | adds                                                  |
| ------------ | ----------------------------------------------------- |
| `[opsin]`    | `py2opsin` for OPSIN-based round-trip verification    |
| `[datasets]` | `datasets` + `tqdm` for PubChem/QM9-style evaluations |
| `[web]`      | FastAPI + uvicorn for the HTTP service                |
| `[dev]`      | pytest, ruff, pre-commit, hypothesis, py2opsin        |

```bash
pip install "bluenamer[opsin,datasets]"
```

## Quick start

```python
from bluenamer import name_smiles

name_smiles("CCO")          # 'ethanol'
name_smiles("c1ccccc1")     # 'benzene'
name_smiles("CC(=O)O")      # 'acetic acid'
```

### Typed result with rules hit + OPSIN round-trip

For everything richer than the bare string, use `bluenamer.name`:

```python
from bluenamer import name

result = name("CC(=O)Nc1ccccc1", include_trace=True, verify_opsin=True)

result.name           # 'N-phenylacetamide'
result.smiles         # 'CC(=O)Nc1ccccc1'
result.ok             # True
result.rules_hit      # ('P-44', 'P-45', 'P-41', 'P-61', 'P-67', ...)
result.rule_hints     # ('Parent hydride / parent structure: Blue Book P-44 ...',)
result.opsin_check.status   # 'matched' | 'mismatched' | 'skipped_no_java' | ...
result.verified       # True when opsin_check is matched
```

Errors do not raise — they are captured on `result.error`, which makes
the batch API safe to point at noisy datasets:

```python
from bluenamer import name_many

results = name_many(
    ["CCO", "c1ccccc1", "definitely-not-a-smiles"],
    processes="auto",       # or an integer, or 1 for in-process
    verify_opsin=False,
)
[r.name for r in results if r.ok]
```

For the full decision trace (one `TraceStep` per phase: parse, perception,
parent selection, numbering, assembly, …):

```python
from bluenamer import analyze_smiles

analysis = analyze_smiles("CC(=O)Nc1ccccc1")
for step in analysis.decisions:
    print(step.phase, step.decision, step.reason)
```

### Natural-language description (`describe`)

`bluenamer.describe(smiles)` walks the same trace and renders a
deterministic, multi-paragraph explanation of how the name is built.
Useful for explainability views and for generating (SMILES, name,
description) training tuples:

```python
from bluenamer import describe

d = describe("CC(=O)Nc1ccccc1")
print(d)            # multi-paragraph prose
d.rules_hit         # ('P-44', 'P-45', 'P-41', 'P-61', 'P-67')
d.components[0]     # DescribedComponent(phase='parse', text='RDKit parsed ...')
```

Same input → same output. No LLM in the loop.

### CLI

```bash
bluenamer name "CC(=O)Nc1ccccc1"            # → N-phenylacetamide
bluenamer name "CC(=O)Nc1ccccc1" --json     # JSON with trace + rules
bluenamer batch smiles.txt --output names.jsonl --processes auto
```

## Development

```bash
git clone https://github.com/lamalab-org/iupac-name-generator
cd iupac-name-generator
pip install -e ".[dev]"

# run the unit + round-trip tests
pytest

# run only fast tests
pytest -m "not slow and not dataset"

# lint and format
ruff check --fix src/bluenamer
ruff format src/bluenamer
```

Java is required for the OPSIN-based round-trip checks (see `py2opsin`).

## HTTP service (Docker)

The `[web]` extra ships a FastAPI app with `name`, `batch`, `describe`
and `healthz` endpoints. The bundled `Dockerfile` includes a Java 17 JRE
so `verify_opsin=True` works out of the box.

```bash
# build + run
docker build -t bluenamer:local .
docker run --rm -p 8000:8000 bluenamer:local

# or via compose
docker compose -f docker/compose.yaml up --build
```

Call the API:

```bash
curl -X POST localhost:8000/name -H 'content-type: application/json' \
     -d '{"smiles":"CC(=O)Nc1ccccc1","include_trace":true,"verify_opsin":true}'

curl -X POST localhost:8000/batch -H 'content-type: application/json' \
     -d '{"smiles":["CCO","c1ccccc1","CC(=O)O"],"processes":1}'

curl -X POST localhost:8000/describe -H 'content-type: application/json' \
     -d '{"smiles":"CC(=O)Nc1ccccc1"}'
```

OpenAPI docs are served at `http://localhost:8000/docs`.

## License

MIT. See `LICENSE`.
