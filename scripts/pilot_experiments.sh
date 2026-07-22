#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# Pre-flight dataset check. anomaly_archive/smd auto-download inside main.py if
# missing, but smap/msl (NASA) and iops have no auto-download and must already
# be in place, so fail fast here instead of mid-run.
missing=0

if [ ! -d "dataset/NASA/train" ] || [ ! -d "dataset/NASA/test" ] || [ ! -f "dataset/NASA/labeled_anomalies.csv" ]; then
  echo "[MISSING] dataset/NASA/ (needed for smap/msl) — run: unzip dataset/NASA.zip -d dataset/"
  missing=1
else
  echo "[OK] dataset/NASA/ found (smap/msl)"
fi

if [ -z "$(ls -A dataset/IOPS/*.train.out 2>/dev/null)" ]; then
  echo "[MISSING] dataset/IOPS/*.train.out (needed for iops) — extract the IOPS/ folder from TSB-UAD-Public.zip into dataset/IOPS/"
  missing=1
else
  echo "[OK] dataset/IOPS/ found (iops)"
fi

echo "[OK] anomaly_archive and smd auto-download on first run if missing (no pre-check needed)"

if [ "$missing" -eq 1 ]; then
  echo "One or more required datasets are missing. Aborting before installing/training."
  exit 1
fi

VENV_DIR="/home/aibiz/setup/volumes/core/core-venv"

# One-time environment setup (idempotent: skipped if VENV_DIR already exists)
if [ ! -d "$VENV_DIR" ]; then
  mkdir -p "$(dirname "$VENV_DIR")"
  python3 -m venv "$VENV_DIR"
fi
. "$VENV_DIR/bin/activate"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

gpu=0
seed=0
pilot_entities=3
epoch=50

for dataset in anomaly_archive iops smd smap msl
do
  echo "${dataset}"
  python main.py --gpu $gpu --dataset $dataset --seed $seed --pilot_entities $pilot_entities --epoch $epoch
done
