"""Read-only system and CUDA/NMS diagnostics."""
import importlib
import platform
import subprocess
import sys
from pathlib import Path

from .common import YOLOV5_COMMIT, require_upstream


def diagnose(require_cuda=False):
    report = {"python": sys.version.split()[0], "platform": platform.platform(), "architecture": platform.machine(),
              "yolov5_expected_commit": YOLOV5_COMMIT, "errors": []}
    release = Path("/etc/nv_tegra_release")
    if release.exists():
        report["l4t"] = release.read_text().strip()
        package = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"], capture_output=True, text=True)
        report["jetpack"] = package.stdout.strip() if package.returncode == 0 else "meta-package not installed; see l4t"
    for name in ("numpy", "PIL", "yaml", "cv2", "torch", "torchvision", "ultralytics", "tensorrt"):
        try:
            module = importlib.import_module(name)
            report[name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            report[name] = f"unavailable: {type(exc).__name__}"
            if name != "tensorrt":
                report["errors"].append(f"Cannot import {name}")
    try:
        import torch
        import torchvision
        report["cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if require_cuda and device != "cuda":
            report["errors"].append("CUDA unavailable; install a platform-compatible GPU PyTorch build")
        if device == "cuda":
            report["gpu"] = torch.cuda.get_device_name(0)
            # Exercise kernels, not just driver enumeration.
            report["cuda_compute"] = float((torch.ones((8, 8), device=device) @ torch.ones((8, 8), device=device)).sum())
        boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]], device=device)
        result = torchvision.ops.nms(boxes, torch.tensor([.9, .8], device=device), .5).cpu().tolist()
        report["nms_device"], report["nms_ok"] = device, result == [0]
        if not report["nms_ok"]:
            report["errors"].append("torchvision NMS returned unexpected result")
    except Exception as exc:
        report["errors"].append(f"Torch/torchvision runtime check failed: {exc}")
    try:
        require_upstream()
    except Exception as exc:
        report["errors"].append(str(exc))
    return report
