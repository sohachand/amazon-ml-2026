#!/bin/bash
# Heavy run (embeddings + cross-encoder + sibling stage) on a LOCAL machine with an NVIDIA GPU.
# Needs: Linux or Windows WSL2 (NOT plain Windows), NVIDIA GPU + driver, Python 3.10-3.12,
#        32 GB+ RAM (64 GB recommended), ~100 GB free disk.
# Usage: put er_gpu_full.zip, stage3_patch.zip and this script in one folder, then
#        bash run_local_heavy.sh
# Resumable: if it stops, run the same command again and it continues where it left off.
set -e
cd "$(dirname "$0")"
nvidia-smi -L || { echo "No NVIDIA GPU / driver found - stop."; exit 1; }
[ -d er_gpu ] || python3 -c "import zipfile; zipfile.ZipFile('er_gpu_full.zip').extractall('.')"
python3 -c "import zipfile; zipfile.ZipFile('stage3_patch.zip').extractall('.')"   # newest code
cd er_gpu
# reuse the machine's own PyTorch if it already sees the GPU (GPU cloud templates); else use a venv
if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  [ -d .venv ] || python3 -m venv .venv
  source .venv/bin/activate
  pip install -q -U pip
fi
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || \
  pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements_gpu.txt
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('GPU:', torch.cuda.get_device_name(0))"
python unpack_data.py
# size the run to the machine
RAM_GB=$(free -g | awk '/Mem:/{print $2}')
export N_PROC=${N_PROC:-$(nproc)} TOKENIZERS_PARALLELISM=false
if [ "${FAST:-0}" = "1" ]; then            # FAST=1: standard settings, ~2-3 h on a 32-CPU machine
  export TFIDF_TOP_N=30 EMB_TOP_N=20 EMB_PAIRS=1000000 CE_PAIRS=1000000; FRAC=0.08
elif [ "$RAM_GB" -ge 60 ]; then
  export TFIDF_TOP_N=50 EMB_TOP_N=30 EMB_PAIRS=2500000 CE_PAIRS=2400000; FRAC=0.15
else
  export TFIDF_TOP_N=30 EMB_TOP_N=20 EMB_PAIRS=1200000 CE_PAIRS=1500000; FRAC=0.08
  [ "$N_PROC" -gt 8 ] && export N_PROC=8     # fewer parallel workers -> less RAM
fi
echo "RAM ${RAM_GB} GB, ${N_PROC} workers, train-frac ${FRAC}"
mkdir -p logs
python src/pipeline.py --data dataset --work work --out output --gpu --ce --train-frac $FRAC 2>&1 | tee -a logs/run.log
python utils/validate_submission.py --matching output/matching_results.tsv --test-dir dataset/test | tail -1
grep -E "FAILED|stage-2      :|stage-3       :|not help" logs/run.log | tail -4
echo "DONE -> $(pwd)/output/matching_results.tsv   (send logs/run.log too)"
