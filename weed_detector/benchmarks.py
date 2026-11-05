"""Separate warmed pipeline timing from labeled-model accuracy checks."""
import json
import time
from pathlib import Path

import numpy as np

from .common import new_run, utc_now, write_json


def latency_stats(milliseconds):
    samples = np.asarray(milliseconds, dtype=float)
    if not len(samples) or not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("Timing samples must be finite and positive")
    mean = float(samples.mean())
    return {"samples": len(samples), "mean_ms": mean, "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)), "pipeline_fps": 1000 / mean}


def benchmark(args):
    from .backend import Detector
    from .dataset import image_files
    from .sources import readable_image
    from .vision import FrameProcessor, restore_detections

    if args.iterations < 1 or args.warmup < 1 or args.images < 1:
        raise ValueError("Iterations, warmup and image count must be positive")
    files = image_files(args.source)[:args.images]
    if not files:
        raise ValueError("No benchmark images found")
    images = [readable_image(path) for path in files]
    model = Detector(args.weights, args.device, args.imgsz, args.conf, .45, args.half, args.data)
    processor = FrameProcessor(args.mirror, args.enhance, args.gamma, not args.no_quality)
    timings, model_times, counts = [], [], []
    for i in range(args.warmup + args.iterations):
        original = images[i % len(images)]
        started = time.perf_counter()
        image, _, _ = processor.apply(original)
        detections, model_ms = model.predict(image)
        detections = restore_detections(detections, original.shape[1], original.shape[0], args.mirror)
        elapsed = (time.perf_counter() - started) * 1000
        if i >= args.warmup:
            timings.append(elapsed)
            model_times.append(model_ms)
            counts.append(len(detections))
    report = {"created_at": utc_now(), "weights": str(Path(args.weights).resolve()), "imgsz": args.imgsz,
              "warmup_iterations": args.warmup, "image_count": len(images), "conf": args.conf,
              "device": str(model.device), "torch": model.torch.__version__, "cuda": model.torch.version.cuda,
              "input_precision": "fp16" if model.model.fp16 else "fp32", "quality_enabled": not args.no_quality,
              "mirror": args.mirror, "enhance": args.enhance, "gamma": args.gamma,
              "pipeline": latency_stats(timings), "model_pipeline": latency_stats(model_times),
              "mean_detections": float(np.mean(counts)),
              "timing_scope": "Warmed batch=1 memory-image pipeline; includes quality/enhancement, letterbox, transfers, model, NMS and box restoration; excludes capture/decode and result writing."}
    if model.device.type == "cuda":
        report["gpu"] = model.torch.cuda.get_device_name(model.device)
    if model.model.engine:
        import tensorrt
        report["tensorrt"] = tensorrt.__version__
    directory = new_run(args.project, args.name)
    write_json(directory / "benchmark.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Benchmark: {directory}")


def accuracy_decision(reference, candidate, max_map_drop, max_recall_drop):
    if not 0 <= max_map_drop <= 1 or not 0 <= max_recall_drop <= 1:
        raise ValueError("Accuracy drop tolerances must be in [0,1]")
    values = [reference[k] for k in ("mAP50_95", "recall")] + [candidate[k] for k in ("mAP50_95", "recall")]
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Accuracy metrics must be finite fractions in [0,1]")
    map_drop = reference["mAP50_95"] - candidate["mAP50_95"]
    recall_drop = reference["recall"] - candidate["recall"]
    return {"mAP50_95_drop": map_drop, "recall_drop": recall_drop,
            "max_map_drop": max_map_drop, "max_recall_drop": max_recall_drop,
            "passed": map_drop <= max_map_drop + 1e-12 and recall_drop <= max_recall_drop + 1e-12}


def compare_models(args):
    from .evaluation import pipeline_metrics
    if Path(args.reference).suffix != ".pt":
        raise ValueError("The FP32 reference must be a YOLOv5 .pt checkpoint")
    if not 0 <= args.max_map_drop <= 1 or not 0 <= args.max_recall_drop <= 1:
        raise ValueError("Accuracy drop tolerances must be in [0,1]")
    directory = new_run(args.project, args.name)
    metrics = {}
    for label, weights in (("reference", args.reference), ("candidate", args.candidate)):
        metrics[label] = pipeline_metrics(weights, args.data, args.split, args.device, args.imgsz, directory / label)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {"reference": str(Path(args.reference).resolve()), "candidate": str(Path(args.candidate).resolve()),
              "split": args.split, "metrics": metrics,
              **accuracy_decision(metrics["reference"], metrics["candidate"], args.max_map_drop, args.max_recall_drop),
              "scope": "Relative mAP/recall regression check only; this does not establish adequate absolute accuracy or real-time performance."}
    write_json(directory / "comparison.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_pass and not report["passed"]:
        raise RuntimeError(f"Accuracy regression exceeded tolerance. See {directory / 'comparison.json'}")
