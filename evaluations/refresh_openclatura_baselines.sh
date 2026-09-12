#!/usr/bin/env bash
# Regenerate every OpenClatura prediction used by the paper-evaluation CI and
# rescore the generated names through OPSIN. Run from any directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-3}"
WORKERS_PER_SHARD="${WORKERS_PER_SHARD:-4}"
OPSIN_WORKERS="${OPSIN_WORKERS:-12}"
STATUS_FILE="${STATUS_FILE:-/private/tmp/openclatura-paper-evaluation-refresh.status}"

cd "$REPO_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable is unavailable: $PYTHON_BIN" >&2
    exit 1
fi

printf 'running\n' > "$STATUS_FILE"
trap 'status=$?; if [[ $status -ne 0 ]]; then printf "failed (%s)\n" "$status" > "$STATUS_FILE"; fi' EXIT

inputs=(
    evaluations/data/pubchem/*_input.jsonl
    evaluations/data/qm9/*_input.jsonl
    evaluations/data/zinc22/*_input.jsonl
)

run_prediction() {
    local input_path="$1"
    local dataset_name stem_name output_path temporary_output
    dataset_name="$(basename "$(dirname "$input_path")")"
    stem_name="$(basename "$input_path" _input.jsonl)"
    output_path="evaluations/results/$dataset_name/${stem_name}_openclatura.jsonl"
    temporary_output="${output_path}.tmp.$$"

    echo "[predict] $input_path -> $output_path"
    PYTHONPATH=src "$PYTHON_BIN" evaluations/predict.py \
        --model openclatura \
        --input "$input_path" \
        --output "$temporary_output" \
        --workers "$WORKERS_PER_SHARD" \
        --chunk 500
    mv "$temporary_output" "$output_path"
}

echo "Regenerating ${#inputs[@]} prediction datasets ($PARALLEL_SHARDS shards in parallel)"
pids=()
for input_path in "${inputs[@]}"; do
    run_prediction "$input_path" &
    pids+=("$!")
    if [[ ${#pids[@]} -eq $PARALLEL_SHARDS ]]; then
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        pids=()
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "Rescoring regenerated predictions with OPSIN"
(
    cd evaluations
    PYTHONWARNINGS=ignore PYTHONPATH=../src "$PYTHON_BIN" score_opsin_std.py \
        results/*/*_openclatura.jsonl \
        --name-key openclatura_iupac \
        --workers "$OPSIN_WORKERS" \
        --chunk 1000
)

printf 'complete\n' > "$STATUS_FILE"
echo "Paper-evaluation baseline refresh complete"
