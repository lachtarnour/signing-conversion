#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/dataset/m4singer.yaml}"

python3 scripts/prepare_m4singer_dataset.py \
    --config "${CONFIG}"

