#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_jetson.sh /absolute/path/data.yaml [extra train arguments]" >&2
  exit 1
fi
DATASET="$1"
shift
.venv/bin/python -m weed_detector train --data "$DATASET" --model yolov5s \
  --device 0 --imgsz 640 --batch 2 --workers 0 --epochs 100 "$@"
