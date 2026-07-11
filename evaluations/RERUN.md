# Rerunning the evals

STOUT vs OpenClatura on QM9, PubChem, and ZINC22. Two models name every SMILES;
each IUPAC name is round-tripped back to SMILES with OPSIN and compared to the
input after full RDKit standardization (Cleanup → normalize → reionize →
uncharge → tautomer-canonicalize). Accuracy = fraction that round-trips to the
same standardized structure.

## Results (standardized + tautomer-canonical OPSIN match)

| dataset | molecules | **OpenClatura** | **STOUT** |
|---------|-----------|-----------------|-----------|
| QM9     | 133,885   | **100.00%**       | 92.55%          |
| PubChem | 5×100,000 | **98.91% ± 0.03** | 97.90% ± 0.04   |
| ZINC22  | 5×100,000 | **96.50% ± 0.07** | 92.27% ± 0.09   |
| **Total** | **1,133,885** | **97.97%** (1,110,918) | **94.78%** (1,074,733) |

Mean ± sample standard deviation across the five 100k seed subsets (PubChem,
ZINC22); QM9 is a single set. OpenClatura wins on every dataset (largest
margins: QM9 +7.5, ZINC22 +4.2), and both models are highly stable across seeds
(std ≤ 0.09).

## Environment (one-time)

```bash
conda env create -f environment.yml         # python 3.11
conda activate stout-pypi-eval
./setup_env.sh                               # openclatura + local batched-GPU STOUT + CUDA 12 wheels
```

`setup_env.sh` installs the **local modified** `STOUT-pypi-2.0.5/` (adds
`translate_forward_batch`, GPU batched decode; output verified identical to the
stock single-item CPU path) as an editable package — so `import STOUT` is the
modified version everywhere. TF 2.15 JIT-compiles H100 (sm_90) kernels from PTX
on first GPU use (one-time warmup).

## Rerun everything

```bash
conda activate stout-pypi-eval
cd evaluations

# 1. Predictions: STOUT (batched GPU, sharded over GPUs 1/2/3) + openclatura (CPU).
#    Writes results/<dataset>/<stem>_<model>.jsonl, one output per line.
#    Re-runnable: existing non-empty outputs are skipped. ~16 h for STOUT.
#    Long-running -> launch inside tmux so it survives disconnect:
tmux new-session -d -s evals 'bash run_all_evals.sh > results/_run_all.log 2>&1'
tmux attach -t evals            # watch; detach with Ctrl-b d

# 2. Score with OPSIN + standardization (fully multicore, ~minutes for 1.1M):
python score_opsin_std.py results/*/*_openclatura.jsonl --name-key openclatura_iupac --workers 64
python score_opsin_std.py results/*/*_stout.jsonl        --name-key stout_iupac       --workers 64
```

Edit the GPU list in `run_all_evals.sh` (`GPUS=(1 2 3)`) to match free GPUs
(`nvidia-smi`).

## Files

| script | role |
|--------|------|
| `predict.py`          | run ONE model on ONE input → one jsonl (`--model stout|openclatura`) |
| `run_all_evals.sh`    | predictions for all datasets, STOUT sharded across GPUs by est. time |
| `score_opsin_std.py`  | OPSIN round-trip + standardized match, multicore; writes summaries + failures |
| `stout_parity.py`     | sanity check: batched GPU STOUT == single-item CPU (50/50 on PubChem) |
| `STOUT-pypi-2.0.5/`   | local modified STOUT (batched GPU decode) |

## Outputs (per input file, in `results/<dataset>/`)

- `<stem>_<model>.jsonl` — predictions: `{index, smiles, <model>_iupac}`
- `<stem>_<model>_opsin_summary.json` — accuracy + counts (`match_method: standardize_and_canonicalize_tautomer`)
- `<stem>_<model>_opsin_failures.csv` — misses with `std_original` / `std_opsin`

Data subsets: `data/{qm9,pubchem,zinc22}/*_input.jsonl` (`{index, smiles}` per line).
