"""Training, evaluation, export, and portable model packaging."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import yaml

from .common import ROOT, UPSTREAM, YOLOV5_COMMIT, import_upstream, new_run, run_upstream, utc_now, write_json
from .dataset import class_names, normalized_config


def train(args):
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"Resume checkpoint not found: {checkpoint}")
        run_upstream("train.py", ["--resume", checkpoint])
        return
    if not args.data:
        raise ValueError("Training requires --data")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.imgsz < 32:
        raise ValueError("Invalid epochs, batch size, workers or image size")
    directory = new_run(args.project, args.name)
    data, report = normalized_config(args.data, directory / "dataset")
    write_json(directory / "dataset-report.json", report)
    write_json(directory / "provenance.json", {"created_at": utc_now(), "yolov5_commit": YOLOV5_COMMIT,
                                               "arguments": {k: v for k, v in vars(args).items() if k != "handler"}})
    weights = Path(args.weights).resolve() if args.weights else ROOT / "weights" / f"{args.model}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    command = ["--data", data, "--weights", "" if args.scratch else weights,
               "--epochs", args.epochs, "--batch-size", args.batch, "--imgsz", args.imgsz,
               "--device", args.device, "--workers", args.workers, "--patience", args.patience,
               "--project", directory.parent, "--name", directory.name, "--exist-ok", "--seed", args.seed]
    if args.scratch:
        command += ["--cfg", UPSTREAM / "models" / f"{args.model}.yaml"]
    if args.hyp:
        hyp = Path(args.hyp).resolve()
        if not hyp.is_file():
            raise ValueError(f"Hyperparameter file not found: {hyp}")
        command += ["--hyp", hyp]
    run_upstream("train.py", command)
    best = directory / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Training returned without best.pt; inspect training logs")
    print(f"Training output: {directory}\nBest checkpoint: {best}")


def evaluate(args):
    if not Path(args.weights).is_file():
        raise ValueError("Evaluation checkpoint does not exist")
    directory = new_run(args.project, args.name)
    data, report = normalized_config(args.data, directory / "dataset")
    if args.split not in report["splits"]:
        raise ValueError(f"Dataset has no {args.split} split")
    import_upstream()
    import val
    results, maps, speeds = val.run(data=str(data), weights=str(Path(args.weights).resolve()),
                                    batch_size=args.batch, imgsz=args.imgsz, device=args.device,
                                    workers=args.workers, task=args.split, project=str(directory.parent),
                                    name=directory.name, exist_ok=True, save_json=False, plots=True)
    names = report["names"]
    metrics = dict(zip(["precision", "recall", "mAP50", "mAP50_95", "val_box_loss", "val_obj_loss", "val_cls_loss"],
                       map(float, results)))
    metrics.update({"split": args.split, "images": report["splits"][args.split]["images"],
                    "per_class_mAP50_95": {names[i]: float(value) for i, value in enumerate(maps)
                                            if report["splits"][args.split]["per_class"].get(i, 0) > 0},
                    "timing_ms": dict(zip(["preprocess", "inference", "nms"], map(float, speeds)))})
    write_json(directory / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Evaluation output: {directory}")


def model_names(weights):
    import_upstream()
    from models.experimental import attempt_load
    model = attempt_load(str(Path(weights).resolve()), device="cpu", fuse=False)
    names = class_names(model.names)
    del model
    return names


def export_model(args):
    weights = Path(args.weights).resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise ValueError("Export requires an existing YOLOv5 .pt checkpoint")
    if args.format == "engine":
        import tensorrt as trt
        if not 8 <= int(trt.__version__.split(".")[0]) <= 10:
            raise ValueError("This exporter supports TensorRT 8-10; use a matching JetPack stack")
        if args.device == "cpu":
            raise ValueError("TensorRT export requires --device 0 on the target Jetson")
    directory = new_run(args.project, args.name)
    copied = directory / "model.pt"
    shutil.copy2(weights, copied)
    names = model_names(weights)
    (directory / "classes.yaml").write_text(yaml.safe_dump({"names": names}, allow_unicode=True), encoding="utf-8")
    command = ["--weights", copied, "--include", args.format, "--imgsz", args.imgsz,
               "--batch-size", 1, "--device", args.device, "--opset", 12, "--workspace", args.workspace]
    if args.half:
        command.append("--half")
    run_upstream("export.py", command)
    artifact = copied.with_suffix(".onnx" if args.format == "onnx" else ".engine")
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise RuntimeError("Export did not produce a valid artifact; upstream may have reported an export error")
    import torch
    environment = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if args.format == "engine":
        environment["tensorrt"] = trt.__version__
        release = Path("/etc/nv_tegra_release")
        environment["l4t"] = release.read_text().strip() if release.is_file() else None
    write_json(directory / "export.json", {"yolov5_commit": YOLOV5_COMMIT, "created_at": utc_now(), **environment,
                                           "imgsz": args.imgsz, "batch": 1, "format": args.format,
                                           "half": args.half, "torch": torch.__version__,
                                           "cuda": torch.version.cuda,
                                           "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
    print(f"Exported model: {artifact}")


def package_model(args):
    weights, output = Path(args.weights).resolve(), Path(args.output).resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise ValueError("Package requires an existing .pt checkpoint")
    if output.exists():
        raise ValueError("Package already exists; choose a new output path")
    names = model_names(weights)
    manifest = {"format_version": 1, "created_at": utc_now(), "yolov5_commit": YOLOV5_COMMIT,
                "file": "best.pt", "names": names, "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(weights, "best.pt")
        archive.writestr("classes.yaml", yaml.safe_dump({"names": names}, allow_unicode=True))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Model package: {output}")
