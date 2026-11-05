import argparse
import json
import logging
import os
import sys

from .common import ROOT, new_run, write_json


def infer(args):
    from .backend import Detector
    from .results import ResultWriter
    from .sources import frames, safe_source
    from .vision import FrameProcessor, restore_detections

    source = args.source or os.environ.get("CAMERA_SOURCE")
    if not source:
        raise ValueError("Provide --source or set CAMERA_SOURCE")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    processor = FrameProcessor(args.mirror, args.enhance, args.gamma, not args.no_quality,
                               args.blur_threshold, args.highlight_threshold)
    detector = Detector(args.weights, args.device, args.imgsz, args.conf, args.iou, args.half, args.data, args.max_det)
    directory = new_run(args.project, args.name)
    write_json(directory / "run.json", {"weights": str(args.weights), "source": safe_source(source),
                                        "imgsz": detector.imgsz, "confidence": args.conf, "iou": args.iou,
                                        "device": args.device, "names": detector.names,
                                        "image_processing": {"mirror": args.mirror, "enhance": args.enhance,
                                                             "gamma": args.gamma, "quality_enabled": not args.no_quality,
                                                             "blur_threshold": args.blur_threshold,
                                                             "highlight_threshold": args.highlight_threshold}})
    iterator = frames(source, args.timeout, args.reconnects)
    writer = ResultWriter(directory, not args.no_visual, args.segment_seconds, args.output_fps)
    status, failure = "completed", None
    print(f"Results: {directory}", flush=True)
    try:
        for frame in iterator:
            image, frame.quality, frame.preprocessing_ms = processor.apply(frame.image)
            detections, milliseconds = detector.predict(image)
            detections = restore_detections(detections, frame.image.shape[1], frame.image.shape[0], args.mirror)
            writer.write(frame, detections, milliseconds)
            if writer.frames == 1 or writer.frames % 100 == 0:
                logging.info("Processed %d frames; current detections=%d, inference=%.1f ms", writer.frames, len(detections), milliseconds)
            if args.max_frames and writer.frames >= args.max_frames:
                break
        if writer.frames == 0:
            raise RuntimeError("Input produced no frames")
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        status, failure = "failed", str(exc)
        raise
    finally:
        iterator.close()
        writer.close(status, failure)


def check_data(args):
    from .dataset import validate_dataset
    _, report, _ = validate_dataset(args.data)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def prepare_data(args):
    from .dataset import prepare_dataset
    print(prepare_dataset(args.images, args.labels, args.output, args.classes, args.seed, args.group_regex))


def doctor(args):
    from .doctor import diagnose
    report = diagnose(args.require_cuda)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, report)
    if report["errors"]:
        raise SystemExit(1)


def augment_data(args):
    from .augmentation import augment_command
    augment_command(args)


def benchmark(args):
    from .benchmarks import benchmark as run
    run(args)


def compare_models(args):
    from .benchmarks import compare_models as run
    run(args)


def stress_test(args):
    from .stress import stress_test as run
    run(args)


def image_processing_arguments(p):
    p.add_argument("--mirror", choices=["none", "horizontal", "vertical", "both"], default="none")
    p.add_argument("--enhance", choices=["none", "clahe", "gamma"], default="none")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma exponent; below 1 brightens")
    p.add_argument("--no-quality", action="store_true", help="Disable per-frame quality diagnostics")


def parser():
    from . import workflows
    root = argparse.ArgumentParser(description="YOLOv5 weed detector: train, evaluate and deploy to Jetson")
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor", help="Inspect JetPack, CUDA, PyTorch and torchvision NMS")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output")
    p.set_defaults(handler=doctor)
    p = sub.add_parser("check-data", help="Validate YOLO detection data and cross-split leakage")
    p.add_argument("--data", required=True)
    p.set_defaults(handler=check_data)
    p = sub.add_parser("prepare-data", help="Copy unsplit images/labels into seeded 80/10/10 splits")
    for name in ("images", "labels", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-regex", help="Regex with a capture group for scene/sequence ID")
    p.set_defaults(handler=prepare_data)
    conditions = ["glare", "gaussian-blur", "motion-blur", "mirror-horizontal", "mirror-vertical", "dark", "jpeg"]
    p = sub.add_parser("augment-data", help="Augment train only; preserve original val/test files and synchronize mirror labels")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--conditions", nargs="+", choices=conditions,
                   default=["glare", "gaussian-blur", "motion-blur", "mirror-horizontal", "dark", "jpeg"])
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--severity", type=float, default=.4)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(handler=augment_data)
    p = sub.add_parser("train", help="Train on PC or Jetson; use batch 2/workers 0 on Jetson initially")
    p.add_argument("--data")
    p.add_argument("--model", choices=["yolov5n", "yolov5s", "yolov5m"], default="yolov5s")
    p.add_argument("--weights")
    p.add_argument("--hyp", help="YOLOv5 training hyperparameters, e.g. configs/hyp.weeds.yaml")
    p.add_argument("--scratch", action="store_true")
    p.add_argument("--resume", help="Resume last.pt using the original run's training settings")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default=str(ROOT / "runs" / "train"))
    p.add_argument("--name", default="weed")
    p.set_defaults(handler=workflows.train)
    p = sub.add_parser("evaluate", help="Evaluate on val or held-out test and save metrics.json")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--project", default=str(ROOT / "runs" / "evaluate"))
    p.add_argument("--name", default="weed")
    p.set_defaults(handler=workflows.evaluate)
    p = sub.add_parser("infer", help="Infer images, video, USB camera index or RTSP")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", help="Path, camera index or stream URL; defaults to CAMERA_SOURCE environment variable")
    p.add_argument("--data", help="Class names YAML; required for TensorRT")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--half", action="store_true")
    p.add_argument("--no-visual", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--reconnects", type=int, default=5)
    p.add_argument("--segment-seconds", type=float, default=60)
    p.add_argument("--output-fps", type=float)
    image_processing_arguments(p)
    p.add_argument("--blur-threshold", type=float, default=60.0)
    p.add_argument("--highlight-threshold", type=float, default=.05)
    p.add_argument("--project", default=str(ROOT / "runs" / "infer"))
    p.add_argument("--name", default="weed")
    p.set_defaults(handler=infer)
    p = sub.add_parser("export", help="Export ONNX or build a TensorRT 8-10 engine on the target device")
    p.add_argument("--weights", required=True)
    p.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--workspace", type=int, default=2, help="TensorRT builder workspace in GiB")
    p.add_argument("--project", default=str(ROOT / "runs" / "export"))
    p.add_argument("--name", default="weed")
    p.set_defaults(handler=workflows.export_model)
    p = sub.add_parser("package-model", help="Package .pt, class names and SHA256 manifest for transfer to Jetson")
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(handler=workflows.package_model)
    p = sub.add_parser("benchmark", help="Measure warmed batch-1 PT/TensorRT pipeline latency, excluding decode/write")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="Image or image directory")
    p.add_argument("--data", help="Class names YAML, required for engines")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--half", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--images", type=int, default=8)
    image_processing_arguments(p)
    p.add_argument("--project", default=str(ROOT / "runs" / "benchmark"))
    p.add_argument("--name", default="weed")
    p.set_defaults(handler=benchmark)
    p = sub.add_parser("compare-models", help="Check precision regression with identical production preprocessing")
    p.add_argument("--reference", required=True, help="Baseline .pt model")
    p.add_argument("--candidate", required=True, help="Candidate .pt or .engine")
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-map-drop", type=float, default=.01)
    p.add_argument("--max-recall-drop", type=float, default=.02)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--project", default=str(ROOT / "runs" / "compare"))
    p.add_argument("--name", default="weed")
    p.set_defaults(handler=compare_models)
    p = sub.add_parser("stress-test", help="Evaluate baseline, glare, blur and mirrors on labeled held-out data")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--conditions", nargs="+", choices=conditions,
                   default=["glare", "gaussian-blur", "motion-blur", "mirror-horizontal"])
    p.add_argument("--severity", type=float, default=.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--project", default=str(ROOT / "runs" / "stress"))
    p.add_argument("--name", default="weed")
    p.set_defaults(handler=stress_test)
    return root


def main():
    # Upstream opens some YAML files using the interpreter's locale encoding.
    # Enforce UTF-8 before it starts, including Windows paths with Chinese names.
    if os.name == "nt" and not sys.flags.utf8_mode:
        import subprocess
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", "weed_detector", *sys.argv[1:]])
        raise SystemExit(result.returncode)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parser().parse_args()
    try:
        args.handler(args)
    except KeyboardInterrupt:
        print("Interrupted; any open result files have been finalized.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
