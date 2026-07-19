# openclatura

**Open Nomenclature Framework**

`openclatura` is a deterministic SMILES-to-IUPAC name generator inspired by the IUPAC Blue Book 2013 recommendations.

Built on top of RDKit, the package walks the molecular graph, detects functional groups and ring systems, selects the principal parent, assigns locants, and constructs the corresponding substitutive IUPAC name. Every step is recorded in an inspectable
decision trace so the *why* of a name is recoverable, not just the *what*.

> **Status:** beta. The naming engine handles a broad slice of organic
> structures (alkanes/alkenes/alkynes, common functional groups, simple
> heterocycles, fused/spiro/bridged systems, retained names from the Blue
> Book). PubChem/QM9/ZINC22 coverage is 99/97/100 %; see `evaluations/` .

## Install

```bash
pip install openclatura
```

Optional extras:

| extra        | adds                                                  |
| ------------ | ----------------------------------------------------- |
| `[opsin]`    | `py2opsin` for OPSIN-based round-trip verification    |
| `[datasets]` | `datasets` + `tqdm` for PubChem/QM9-style evaluations |
| `[web]`      | FastAPI + uvicorn for the HTTP service                |
| `[dev]`      | pytest, ruff, pre-commit, hypothesis, py2opsin        |

```bash
pip install "openclatura[opsin,datasets]"
```

The default install does **not** include OPSIN verification. Install the
`[opsin]` extra and make sure Java 8+ is available if you want round-trip
verification through OPSIN:

```bash
pip install "openclatura[opsin]"
java -version
```

If `py2opsin` is installed but Java is missing or inaccessible, OPSIN
verification is skipped gracefully. The name generation still succeeds and the
verification status is reported as `skipped_no_java`.

## Quick start

```python
from openclatura import name_smiles

name_smiles("CCO")          # 'ethanol'
name_smiles("c1ccccc1")     # 'benzene'
name_smiles("CC(=O)O")      # 'acetic acid'
```

### Typed result with rules hit + OPSIN round-trip

For everything richer than the bare string, use `openclatura.name`:

```python
from openclatura import name

result = name("CC(=O)Nc1ccccc1", include_trace=True, verify_opsin=True)

result.name           # 'N-phenylacetamide'
result.smiles         # 'CC(=O)Nc1ccccc1'
result.ok             # True
result.rules_hit      # ('P-44', 'P-45', 'P-41', 'P-61', 'P-67', ...)
result.rule_hints     # ('Parent hydride / parent structure: Blue Book P-44 ...',)
result.opsin_check.status   # 'matched' | 'mismatched' | 'skipped_no_java' | ...
result.verified       # True when opsin_check is matched
```

`verify_opsin` defaults to `False`. When set to `True`, verification is
best-effort and does not raise if OPSIN support is unavailable:

- no `py2opsin` installed: `result.opsin_check.status == "skipped_no_opsin"`
- `py2opsin` installed but Java unavailable: `status == "skipped_no_java"`
- OPSIN parses and round-trips: `status == "matched"` or `"mismatched"`

Errors do not raise — they are captured on `result.error`, which makes
the batch API safe to point at noisy datasets:

```python
from openclatura import name_many

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
from openclatura import analyze_smiles

analysis = analyze_smiles("CC(=O)Nc1ccccc1")
for step in analysis.decisions:
    print(step.phase, step.decision, step.reason)
```
### CLI

```bash
openclatura name "CC(=O)Nc1ccccc1"            # → N-phenylacetamide
openclatura name "CC(=O)Nc1ccccc1" --json     # JSON with trace + rules
openclatura batch smiles.txt --output names.jsonl --processes auto
```

The CLI verifies with OPSIN by default when possible. This is different from the
Python API, where `verify_opsin=False` by default. Disable CLI verification with
`--no-verify`:

```bash
openclatura name "CC(=O)Nc1ccccc1" --no-verify
```

If OPSIN support is unavailable, the command still prints the generated name and
reports the verification status:

```text
N-phenylacetamide
  opsin: skipped_no_java
```

Other possible skipped statuses include `skipped_no_opsin` when `py2opsin` is
not installed. Install `openclatura[opsin]` and Java 8+ for full CLI
verification.

### Natural-language description (`describe`)

`openclatura.describe(smiles)` walks the same trace and renders a
deterministic, multi-paragraph explanation of how the name is built.
Useful for explainability views and for generating (SMILES, name,
description) training tuples:

```python
from openclatura import describe

d = describe("CC(=O)Nc1ccccc1")
print(d)            # multi-paragraph prose
d.rules_hit         # ('P-44', 'P-45', 'P-41', 'P-61', 'P-67')
d.components[0]     # DescribedComponent(phase='parse', text='RDKit parsed ...')
```

Same input → same output. No LLM in the loop.

## Human-like description

Openclatura can generate uncanny human-like descriptions of molecules.
```python

from openclatura import describe_human

d = describe_human("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
print(d.text)

""" Input SMILES: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Processed SMILES: Cn1cnc2c1c(=O)n(C)c(=O)n2C
Atom ids in that SMILES: C{0}n{1}1c{2}n{3}c{4}2c{5}1c{6}(=O{7})n{8}(C{13})c{9}(=O{10})n{11}2C{12}

The molecule is named 1,3,7-trimethylpurine-2,6-dione.

The molecule is built around the retained purine parent, 9-membered bicyclic [4.3.0] heteroskeleton.
Within that parent framework, there is nitrogen at positions 1 (atom id 8), 3 (atom id 11), 7 (atom id 1), and 9 (atom id 3).
The principal characteristic feature is oxo groups at positions 2 (atom id 9) and 6 (atom id 6).
Attached to this framework are methyl groups at positions 1 (atom id 8), 3 (atom id 11), and 7 (atom id 1). """

```

## Development

```bash
git clone https://github.com/lamalab-org/iupac-name-generator
cd iupac-name-generator
pip install -e ".[dev]"

# run the unit + round-trip tests
pytest

# run only fast tests
pytest -m "not slow and not dataset and not golden"

# strict RDKit-version regression suite (also runs in the rdkit-compat CI job)
pytest -m golden

# lint and format
ruff check --fix src/openclatura
ruff format src/openclatura
```

Java is required for the OPSIN-based round-trip checks (see `py2opsin`).

## HTTP service (Docker)

The `[web]` extra ships a FastAPI app with `name`, `batch`, `describe`
and `healthz` endpoints. The bundled `Dockerfile` includes a headless JRE
so `verify_opsin=True` works out of the box.

```bash
# build + run
docker build -t openclatura:local .
docker run --rm -p 8000:8000 openclatura:local

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

## How to cite

If you use Openclatura in your research, please cite the Openclatura preprint:

```bibtex
@article{openclatura2026,
  author  = {Mirza, Adrian and Jablonka, Kevin Maik and Fedorov, Rostislav},
  title   = {Openclatura--An Open-Source Nomenclature Framework for Rule-Based Molecule Naming},
  journal = {ChemRxiv},
  year    = {2026},
  month   = jul,
  day     = {15},
  doi     = {10.26434/chemrxiv.15006114/v1},
  url     = {https://doi.org/10.26434/chemrxiv.15006114/v1},
  note    = {Preprint}
}
```

If you are using OPSIN for verification, please cite the original OPSIN publication:

```bibtex
@article{lowe2011opsin,
  author  = {Lowe, Daniel M. and Corbett, Peter T. and Murray-Rust, Peter and Glen, Robert C.},
  title   = {Chemical Name to Structure: {OPSIN}, an Open Source Solution},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2011},
  volume  = {51},
  number  = {3},
  pages   = {739--753},
  doi     = {10.1021/ci100384d},
  url     = {https://doi.org/10.1021/ci100384d}
}
```