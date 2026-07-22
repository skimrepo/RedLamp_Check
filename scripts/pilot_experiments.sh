#!/bin/bash
set -e

cd "$(dirname "$0")/.."

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
epoch=5

for dataset in anomaly_archive iops smd smap msl
do
  echo "${dataset}"
  python main.py --gpu $gpu --dataset $dataset --seed $seed --pilot_entities $pilot_entities --epoch $epoch
done
