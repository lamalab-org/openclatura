# evaluations

STOUT-vs-OpenClatura evaluation on QM9, ZINC22, and PubChem, plus a parity
check that the repo's locally-modified STOUT still reproduces the upstream
PyPI STOUT outputs.

For every molecule, each namer maps SMILES to an IUPAC name; the name is then
round-tripped back to SMILES through OPSIN and compared (as canonical RDKit
SMILES) with the input. Accuracy is the fraction of molecules whose
OPSIN-reconstructed structure matches the original.

## Layout

```
evaluations/
├── environment.yml     # conda base (python 3.11 + pip)
├── setup_env.sh        # pip installs (openclatura + STOUT-pypi + runtime)
├── backends.py         # openclatura / stout name(smiles)->str loaders
├── evaluate.py         # main: name both backends + OPSIN round-trip + compare
├── stout_parity.py     # modified-vendored STOUT vs PyPI STOUT output diff
├── data/               # input SMILES subsets (index, smiles per line)
│   ├── qm9/            # qm9_all_input.jsonl (368 mols)
│   ├── zinc22/         # 5 × 100k seeded subsets + manifest
│   └── pubchem/        # 5 × 100k seeded subsets + manifest
└── results/            # reference outputs (see "Results" below)
    ├── qm9/            # full run produced by evaluate.py (both backends)
    ├── pubchem/        # existing seed-42 outputs + all-seed summaries
    ├── zinc22/         # existing seed-42 STOUT output + summaries
    └── parity/         # stout_parity.py output
```

## Environment

openclatura requires Python ≥ 3.11; STOUT-pypi 2.0.5 pins `tensorflow==2.10.1`,
which has no Python 3.11 build. The STOUT SavedModel runs correctly on
TensorFlow 2.15.0 — the same version the existing `evals/stout` virtualenv uses
— so we install STOUT-pypi with `--no-deps` and pin TF 2.15.0 ourselves. This
keeps the environment aligned with `evals/stout` so the parity check is
meaningful. (`pip check` will warn about the unmet TF pin; that is expected.)

```bash
conda env create -f environment.yml
conda activate stout-pypi-eval
./setup_env.sh
```

Installed versions: openclatura 0.1.2, STOUT-pypi 2.0.5, tensorflow 2.15.0,
keras 2.15.0, numpy 1.26.4, rdkit 2026.3.3, py2opsin 1.2.0, jpype1 1.7.1.

## Running the evaluation

`evaluate.py` names an input subset with one or both backends, writes a JSONL
per backend (input fields preserved, plus the IUPAC name), OPSIN-validates it,
and writes a failures CSV + summary JSON per backend.

```bash
conda activate stout-pypi-eval

# QM9 (small) — both backends, with OPSIN round-trip
python evaluate.py --input data/qm9/qm9_all_input.jsonl \
    --outdir results/qm9 --backend both

# One PubChem shard — STOUT only
python evaluate.py --input data/pubchem/pubchem_seed42_100000_input.jsonl \
    --outdir results/pubchem --backend stout

# One PubChem shard — openclatura only (writes the openblue_iupac key,
# reproducing the legacy OpenBlue result format)
python evaluate.py --input data/pubchem/pubchem_seed42_100000_input.jsonl \
    --outdir results/pubchem --backend openclatura
```

Output file names follow `{input-stem-without-_input}_{backend}.jsonl`,
`..._{backend}_opsin_failures.csv`, `..._{backend}_opsin_summary.json`, plus a
combined `..._comparison.json`. The JSONL schema and summary schema match the
existing files under `evals/` — the openclatura backend defaults to the
`openblue_iupac` key so its output is drop-in compatible with the legacy
`*_openblue.jsonl` files (the file-name label is `openclatura` rather than the
old `openblue`).

Run the full 5×100k subsets by pointing `--input` at each seeded file under
`data/pubchem/` and `data/zinc22/`.

## Batched GPU STOUT

`STOUT-pypi-2.0.5/` is a local copy of the stock PyPI package with a batched
GPU decoder added (`translate_forward_batch`, backed by a `tf.while_loop`
greedy decode in `STOUT/stout.py`). The stock package only exposes single-item
`translate_forward`; the SavedModel's export signature is hard-pinned to batch
size 1, so batching drives the SavedModel's own trained transformer directly.

Run it with `stout_batch_gpu.py` (honors a pre-set `CUDA_VISIBLE_DEVICES`):

```bash
conda activate stout-pypi-eval
CUDA_VISIBLE_DEVICES=1 python stout_batch_gpu.py \
    --input data/qm9/qm9_all_input.jsonl \
    --output results/qm9/qm9_all_stout_batch.jsonl --batch-size 128
```

Throughput is ~5–6 mol/s per H100 (vs ~0.1/s single-item CPU). Notes/limits:
batch size is capped around **256** — larger batches overflow the encoder
self-attention softmax kernel (`[B, 8, 602, 602]`). TF 2.15 also ships no native
H100 (sm_90) kernels, so kernels JIT-compile from PTX on first use.

### Compatibility: CPU single vs GPU batched

`stout_parity.py` confirms the batched GPU path returns exactly the same names
as the stock single-item `translate_forward` on CPU (both from this same local
package):

```bash
python stout_parity.py compare \
    --input data/pubchem/pubchem_seed42_100000_input.jsonl \
    --outdir results/parity --limit 50 --gpu 1
```

**Result: 50/50 identical** on a random PubChem sample (`results/parity/`) —
including the small molecules where an earlier modified STOUT fork regressed
(`acetamide`, `urea`, `formamide`). The batched decoder is a faithful,
faster drop-in for the single-item version. (The previously-vendored modified
STOUT fork, which regressed accuracy to ~50–73%, has been removed; only
`STOUT-pypi-2.0.5` is used now.)

## Results

`results/` holds reference outputs so the folder is self-contained:

- **qm9/** — a genuine full run of `evaluate.py` over all 368 QM9 molecules
  (both backends), demonstrating the script reproduces the expected file
  formats.
- **pubchem/** — the existing seed-42 STOUT and OpenBlue outputs (full
  100k JSONL + failures CSV + summary) plus the compact `*_opsin_summary.json`
  for all five seeds.
- **zinc22/** — the existing seed-42 STOUT output and eval summaries.
- **parity/** — CPU-single vs GPU-batched compatibility on 50 random PubChem
  molecules (50/50 identical).

The full 5×100k STOUT/OpenBlue outputs for every seed live under
`evals/pubchem_stout_5x100k/`, `evals/pubchem_openblue_5x100k/`, and
`evals/zinc22_stout_eval/`; only the seed-42 reference is copied here to keep
this folder from duplicating ~660 MB.
