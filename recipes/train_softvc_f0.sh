#!/usr/bin/env bash
set -euo pipefail

python -m svc.cli.prepare --config configs/prepare/manifest.yaml
python -m svc.cli.train --config configs/train/softvc_f0.yaml
