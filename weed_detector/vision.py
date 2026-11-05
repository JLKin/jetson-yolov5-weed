"""Measured image quality and optional preprocessing; never invent lost detail."""
import math
import time

import cv2
import numpy as np


def flip_labels(labels, horizontal=False, vertical=False):
    return [(category, 1 - x if horizontal else x, 1 - y if vertical else y, width, height)
            for category, x, y, width, height in labels]


def restore_detections(detections, width, height, mirror):
    """Map continuous xyxy pixel boundaries back to the original input frame."""
    restored = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        if mirror in ("horizontal", "both"):
            x1, x2 = width - x2, width - x1
        if mirror in ("vertical", "both"):
            y1, y2 = height - y2, height - y1
        restored.append({**detection, "bbox_xyxy": [x1, y1, x2, y2]})
    return restored


def quality_metrics(bgr, blur_threshold=60.0, highlight_threshold=0.05):
    """Heuristics at a bounded analysis resolution, not calibrated probabilities."""
    scale = min(1.0, 640 / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    highlights = float(np.mean(np.min(small, axis=2) >= 245))
    dark = float(np.mean(gray <= 15))
    flags = []
    if sharpness < blur_threshold:
        flags.append("blur_risk")
    if highlights > highlight_threshold:
        flags.append("highlight_risk")
    if dark > 0.5:
        flags.append("dark_risk")
    return {"sharpness_laplacian": round(sharpness, 3), "highlight_fraction": round(highlights, 6),
            "dark_fraction": round(dark, 6), "analysis_width": small.shape[1],
            "analysis_height": small.shape[0], "flags": flags}


class FrameProcessor:
    def __init__(self, mirror="none", enhance="none", gamma=1.0, quality=True,
                 blur_threshold=60.0, highlight_threshold=0.05):
        if mirror not in ("none", "horizontal", "vertical", "both") or enhance not in ("none", "clahe", "gamma"):
            raise ValueError("Unsupported mirror or enhancement mode")
        if not math.isfinite(gamma) or not 0.2 <= gamma <= 5 or not math.isfinite(blur_threshold) or blur_threshold < 0:
            raise ValueError("Gamma must be in [0.2,5] and blur threshold nonnegative")
        if not 0 <= highlight_threshold <= 1:
            raise ValueError("Highlight threshold must be in [0,1]")
        self.mirror, self.enhance, self.gamma, self.quality = mirror, enhance, gamma, quality
        self.blur_threshold, self.highlight_threshold = blur_threshold, highlight_threshold
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if enhance == "clahe" else None
        self.lut = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)

    def apply(self, image):
        started = time.perf_counter()
        report = quality_metrics(image, self.blur_threshold, self.highlight_threshold) if self.quality else {"flags": [], "disabled": True}
        transformed = image
        if self.mirror != "none":
            transformed = cv2.flip(transformed, {"horizontal": 1, "vertical": 0, "both": -1}[self.mirror])
        if self.enhance == "clahe":
            lab = cv2.cvtColor(transformed, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            transformed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif self.enhance == "gamma":
            transformed = cv2.LUT(transformed, self.lut)
        report.update({"mirror": self.mirror, "enhance": self.enhance})
        return transformed, report, (time.perf_counter() - started) * 1000
