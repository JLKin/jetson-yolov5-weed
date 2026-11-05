#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -f /etc/nv_tegra_release ]]; then
  echo "This installer is for NVIDIA Jetson. See docs/windows.md for PC setup." >&2
  exit 1
fi
# Ignore an active Conda base environment. NVIDIA apt modules use system Python.
JETSON_PYTHON="${JETSON_PYTHON:-/usr/bin/python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$JETSON_PYTHON" -m venv --system-site-packages .venv
fi
.venv/bin/python scripts/install_runtime.py --jetson
echo "Ready. Activate with: source .venv/bin/activate"
