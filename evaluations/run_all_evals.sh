#!/usr/bin/env bash
# Launch predictions for all evals: STOUT (batched GPU, sharded over GPUs 1/2/3)
# and openclatura (CPU, multiprocess). Predictions only -- one jsonl per
# input x model. Re-runnable: existing non-empty outputs are skipped.
#
#   ./run_all_evals.sh            # run everything
#
# Outputs: results/<dataset>/<stem>_<model>.jsonl  (stem = input minus _input)
set -uo pipefail
cd "$(dirname "$0")"

source /data/mirzaa/miniconda3/etc/profile.d/conda.sh
conda activate stout-pypi-eval

BATCH=256
GPUS=(1 2 3)
LOGDIR=results/_logs
mkdir -p "$LOGDIR" results/qm9 results/pubchem results/zinc22

# Collect input files (dataset dir name is used for the output subdir).
INPUTS=()
for ds in qm9 pubchem zinc22; do
  for f in data/$ds/*_input.jsonl; do
    [ -e "$f" ] && INPUTS+=("$ds|$f")
  done
done

out_path() {  # dataset, input -> results/<ds>/<stem>_<model>.jsonl
  local ds="$1" inp="$2" model="$3"
  local stem; stem="$(basename "$inp" .jsonl)"; stem="${stem%_input}"
  echo "results/$ds/${stem}_${model}.jsonl"
}
nonempty() { [ -s "$1" ]; }

echo "=== Phase 1: openclatura (CPU, background) ==="
(
  for entry in "${INPUTS[@]}"; do
    ds="${entry%%|*}"; inp="${entry#*|}"
    out="$(out_path "$ds" "$inp" openclatura)"
    if nonempty "$out"; then echo "skip $out"; continue; fi
    echo ">> openclatura $inp -> $out"
    python predict.py --model openclatura --input "$inp" --output "$out" --workers 64 \
      >"$LOGDIR/$(basename "$out").log" 2>&1
  done
  echo "openclatura: ALL DONE"
) &
echo "openclatura launched (pid $!)"

echo "=== Phase 2: STOUT (GPUs ${GPUS[*]}, background) ==="
# Greedy-assign input files to GPUs by estimated decode time (rows / per-dataset
# rate) so no GPU straggles. Measured rates: qm9 ~21/s, pubchem ~7.4/s, zinc ~6/s.
declare -a SHARD
assign=$(python - "${#GPUS[@]}" <<'PY'
import sys, glob, os
ngpu = int(sys.argv[1])
rate = {"qm9": 21.0, "pubchem": 7.4, "zinc22": 6.0}
items = []
for ds in ["qm9", "pubchem", "zinc22"]:
    for f in sorted(glob.glob(f"data/{ds}/*_input.jsonl")):
        rows = sum(1 for _ in open(f))
        items.append((rows / rate[ds], f))
items.sort(reverse=True)  # heaviest first
load = [0.0] * ngpu
binof = {}
for cost, f in items:
    g = min(range(ngpu), key=lambda k: load[k])
    load[g] += cost
    binof.setdefault(g, []).append(f)
# index files in the original INPUTS order so bash can map them
allf = []
for ds in ["qm9", "pubchem", "zinc22"]:
    allf += sorted(glob.glob(f"data/{ds}/*_input.jsonl"))
for g in range(ngpu):
    idxs = [str(allf.index(f)) for f in binof.get(g, [])]
    print(f"{g} {' '.join(idxs)}")
sys.stderr.write("est GPU hours: " + ", ".join(f"{h/3600:.1f}" for h in load) + "\n")
PY
)
while read -r g rest; do SHARD[$g]="$rest"; done <<< "$assign"

for g in "${!GPUS[@]}"; do
  gpu="${GPUS[$g]}"
  files="${SHARD[$g]:-}"
  (
    for i in $files; do
      entry="${INPUTS[$i]}"; ds="${entry%%|*}"; inp="${entry#*|}"
      out="$(out_path "$ds" "$inp" stout)"
      if nonempty "$out"; then echo "[gpu$gpu] skip $out"; continue; fi
      echo "[gpu$gpu] stout $inp -> $out"
      CUDA_VISIBLE_DEVICES="$gpu" python predict.py --model stout \
        --input "$inp" --output "$out" --batch-size "$BATCH" \
        >"$LOGDIR/$(basename "$out").log" 2>&1
    done
    echo "[gpu$gpu] ALL DONE"
  ) &
  echo "gpu$gpu launched (pid $!) files:$files"
done

wait
echo "=== ALL EVALS COMPLETE ==="
