#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# One-time environment setup (idempotent: skipped if .venv already exists)
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

gpu=0
seed=0
pilot_entities=3
epoch=5

for dataset in anomaly_archive iops smd smap msl
do
  echo "${dataset}"
  python main.py --gpu $gpu --dataset $dataset --seed $seed --pilot_entities $pilot_entities --epoch $epoch
done
