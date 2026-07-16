#!/usr/bin/env bash
# Install everything for the STOUT-vs-OpenClatura evaluation into the active
# conda environment (create it first with `conda env create -f environment.yml
# && conda activate stout-pypi-eval`). Run from the evaluations/ directory.
#
# Why the local, --no-deps STOUT install:
#   openclatura requires Python >=3.11. Upstream STOUT-pypi 2.0.5 pins
#   tensorflow==2.10.1, which has no build for Python 3.11, so a naive
#   `pip install STOUT-pypi openclatura` cannot resolve. We instead install the
#   *local, modified* STOUT (./STOUT-pypi-2.0.5, which adds batched GPU decoding
#   via translate_forward_batch) as an editable package with --no-deps, and
#   provide tensorflow 2.15.0 + the H100 (sm_90) CUDA wheels ourselves. The
#   batched output is verified identical to the stock single-item CPU path
#   (see stout_parity.py). `pip check` will warn about the unmet
#   tensorflow==2.10.1 pin from STOUT's metadata; that warning is expected.
set -euo pipefail
cd "$(dirname "$0")"

# our namer + eval tooling
pip install --no-input openclatura==0.1.2 py2opsin==1.2.0 datasets tqdm

# the LOCAL modified STOUT (batched GPU) as editable, without its tensorflow pin.
# This is the STOUT used by predict.py / stout_parity.py / the whole eval.
pip install --no-input --no-deps -e ./STOUT-pypi-2.0.5

# STOUT runtime (TF 2.15) + CPU deps
pip install --no-input \
  "tensorflow==2.15.0" \
  "keras==2.15.0" \
  "numpy==1.26.4" \
  pystow==0.8.21 \
  unicodedata2==17.0.1 \
  jpype1==1.7.1

# GPU: TF 2.15 needs the CUDA 12 runtime wheels to see the GPUs (the plain
# tensorflow wheel is CPU-only). These match the versions on the box; on H100
# (sm_90) TF 2.15 has no native kernels and JIT-compiles from PTX on first use.
pip install --no-input \
  nvidia-cublas-cu12==12.2.5.6 nvidia-cuda-cupti-cu12==12.2.142 \
  nvidia-cuda-nvcc-cu12==12.2.140 nvidia-cuda-nvrtc-cu12==12.2.140 \
  nvidia-cuda-runtime-cu12==12.2.140 nvidia-cudnn-cu12==8.9.4.25 \
  nvidia-cufft-cu12==11.0.8.103 nvidia-curand-cu12==10.3.3.141 \
  nvidia-cusolver-cu12==11.5.2.141 nvidia-cusparse-cu12==12.1.2.141 \
  nvidia-nccl-cu12==2.16.5 nvidia-nvjitlink-cu12==12.2.140

echo
echo "Installed. Quick check:"
python - <<'PY'
from openclatura import name_smiles
import STOUT
print("openclatura:", name_smiles("CCO"), "|", name_smiles("c1ccccc1"))
print("STOUT:", STOUT.__file__, "| batched:", hasattr(STOUT, "translate_forward_batch"))
print("GPU:", STOUT.get_device_info().get("physical_gpus"))
PY
echo "STOUT will download its ~1GB model on first call."
