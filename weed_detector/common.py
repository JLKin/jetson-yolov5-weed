"""Paths and explicit, pinned upstream access."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "yolov5"
YOLOV5_COMMIT = "9777cd00e86382dfa5a12e6244ec4f609bc4774c"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def upstream_env():
    return dict(os.environ, YOLOv5_AUTOINSTALL="false", YOLO_AUTOINSTALL="false",
                YOLO_CONFIG_DIR=str(ROOT / "work" / "ultralytics"),
                PYTHONUTF8="1", WANDB_MODE="disabled", COMET_MODE="DISABLED")


def require_upstream():
    if not (UPSTREAM / "train.py").is_file():
        raise ValueError("YOLOv5 is missing. Run: git submodule update --init --recursive")
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != YOLOV5_COMMIT:
        raise ValueError("YOLOv5 revision mismatch. Run: git submodule update --init --recursive")
    return UPSTREAM


def import_upstream():
    require_upstream()
    for key, value in upstream_env().items():
        if key.startswith(("YOLO", "WANDB", "COMET")):
            os.environ[key] = value
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def run_upstream(script, arguments):
    require_upstream()
    command = [sys.executable, str(UPSTREAM / script), *map(str, arguments)]
    subprocess.run(command, cwd=ROOT, env=upstream_env(), check=True)


def new_run(project, name):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Run name must be a single directory name")
    parent = Path(project).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        path = parent / (name if index == 0 else f"{name}_{index:03d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise ValueError("Too many runs with this name")
