#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -z "${CAMERA_SOURCE:-}" ]]; then
  echo 'Set CAMERA_SOURCE to the verified RTSP URL. See docs/camera.md.' >&2
  exit 1
fi
.venv/bin/python -m weed_detector infer --weights weights/best.pt --device 0 "$@"
