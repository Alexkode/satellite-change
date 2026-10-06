#!/bin/bash
set -euo pipefail
python -m change_lab.data
python -m change_lab.experiment --model difference
for model in unet siamese; do
  for seed in 0 1 2; do
    python -m change_lab.experiment --model "$model" --seed "$seed"
  done
done
python scripts/report.py
