"""Install app dependencies without changing the platform GPU framework."""
import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pip(*arguments):
    subprocess.run([sys.executable, "-m", "pip", *arguments], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jetson", action="store_true")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this installer inside the project's virtual environment")
    try:
        import torch
        import torchvision
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]], device="cuda"), torch.tensor([.9], device="cuda"), .5)
    except Exception as exc:
        raise SystemExit(f"Install matching GPU torch and torchvision first: {exc}\nSee docs/jetson.md or docs/windows.md")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision")}
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=ROOT, check=True)
    command = ["install", "-r", str(ROOT / "requirements" / "runtime.txt")]
    if args.jetson:
        l4t = Path("/etc/nv_tegra_release").read_text()
        constraint = "jetpack5-constraints.txt" if "R35" in l4t else "jetson-constraints.txt"
        command += ["-c", str(ROOT / "requirements" / constraint)]
    pip(*command)
    if not args.jetson:
        pip("install", "opencv-python>=4.8,<6")
    pip("install", "--no-deps", "-r", str(ROOT / "requirements" / "helpers.txt"))
    pip("install", "--no-deps", "-e", str(ROOT))
    if versions != {name: importlib.metadata.version(name) for name in versions}:
        raise SystemExit("GPU framework changed unexpectedly; inspect installed packages")
    subprocess.run([sys.executable, "-m", "weed_detector", "doctor", "--require-cuda"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
