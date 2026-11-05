"""Controlled accuracy evaluation using the exact production preprocessing."""
import numpy as np

from .common import write_json
from .dataset import label_path, read_labels, validate_dataset
from .sources import readable_image


def pipeline_metrics(weights, config, split, device, imgsz, output):
    from .backend import Detector
    data, report, files = validate_dataset(config)
    if split not in files or report["splits"][split]["objects"] == 0:
        raise ValueError(f"{split} requires annotated objects for accuracy evaluation")
    # Both backends use batch 1 and the same square letterbox; PT is FP32.
    model = Detector(weights, device, imgsz, confidence=.001, iou=.6, half=False, data=config)
    names = [model.names[i] for i in range(len(model.names))]
    if names != data["names"]:
        raise ValueError("Model class names/order do not match the dataset")
    import torch
    from val import process_batch
    from utils.metrics import ap_per_class
    stats, timings = [], []
    thresholds = torch.linspace(.5, .95, 10)
    model.predict(readable_image(files[split][0]))
    for path in files[split]:
        image = readable_image(path)
        detections, elapsed = model.predict(image)
        height, width = image.shape[:2]
        labels = read_labels(label_path(path), len(data["names"]))
        targets = torch.tensor([[category, (x - w / 2) * width, (y - h / 2) * height,
                                 (x + w / 2) * width, (y + h / 2) * height]
                                for category, x, y, w, h in labels], dtype=torch.float32).reshape(-1, 5)
        predictions = torch.tensor([[*d["bbox_xyxy"], d["confidence"], d["class_id"]]
                                    for d in detections], dtype=torch.float32).reshape(-1, 6)
        correct = process_batch(predictions, targets, thresholds)
        stats.append((correct.numpy(), predictions[:, 4].numpy(), predictions[:, 5].numpy(), targets[:, 0].numpy()))
        timings.append(elapsed)
    combined = [np.concatenate(values, axis=0) for values in zip(*stats)]
    _, _, precision, recall, _, ap, ids = ap_per_class(*combined, plot=False, names=data["names"])
    result = {"precision": float(precision.mean()), "recall": float(recall.mean()),
              "mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean()),
              "images": len(files[split]), "objects": report["splits"][split]["objects"],
              "per_class_mAP50_95": {data["names"][int(category)]: float(ap[i].mean()) for i, category in enumerate(ids)},
              "mean_inference_ms": float(np.mean(timings)), "split": split,
              "protocol": "production square letterbox, batch=1, rounded original-frame boxes, conf=0.001, NMS IoU=0.6, AP IoU=0.5:0.95; PT FP32 and engine native precision"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", result)
    return result
