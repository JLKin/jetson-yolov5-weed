"""Deterministic optical stress fixtures and train-only dataset augmentation."""
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .common import write_json
from .dataset import label_path, read_labels, validate_dataset
from .sources import readable_image
from .vision import flip_labels

CONDITIONS = ("glare", "gaussian-blur", "motion-blur", "mirror-horizontal", "mirror-vertical", "dark", "jpeg")


def transform(image, labels, condition, severity=0.4, seed=42):
    if condition not in CONDITIONS or not 0 < severity <= 1:
        raise ValueError("Unknown condition or severity outside (0,1]")
    rng = np.random.default_rng(seed)
    labels = list(labels)
    if condition.startswith("mirror"):
        horizontal = condition == "mirror-horizontal"
        return cv2.flip(image, 1 if horizontal else 0), flip_labels(labels, horizontal, not horizontal)
    if condition == "gaussian-blur":
        result = cv2.GaussianBlur(image, (0, 0), sigmaX=0.4 + severity * 2.5)
    elif condition == "motion-blur":
        size = 3 + 2 * int(round(severity * 5))
        kernel = np.zeros((size, size), np.float32)
        center = (size - 1) / 2
        angle = rng.uniform(0, np.pi)
        dx, dy = center * np.cos(angle), center * np.sin(angle)
        cv2.line(kernel, (round(center - dx), round(center - dy)), (round(center + dx), round(center + dy)), 1.0, 1)
        kernel /= kernel.sum()
        result = cv2.filter2D(image, -1, kernel)
    elif condition == "glare":
        # An additive elliptical highlight approximates one failure mode of glare.
        # This does not simulate polarization, lens optics or recover real highlights.
        height, width = image.shape[:2]
        yy, xx = np.ogrid[:height, :width]
        x, y = rng.uniform(.15, .85) * width, rng.uniform(.15, .85) * height
        sx, sy = width * (.06 + .12 * severity), height * (.06 + .12 * severity)
        glow = np.exp(-.5 * (((xx - x) / sx) ** 2 + ((yy - y) / sy) ** 2))
        result = np.clip(image.astype(np.float32) + glow[:, :, None] * (60 + 180 * severity), 0, 255).astype(np.uint8)
    elif condition == "dark":
        result = np.clip((image.astype(np.float32) / 255) ** (1 + 1.5 * severity) * 255, 0, 255).astype(np.uint8)
    else:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, round(90 - 65 * severity)])
        if not ok:
            raise RuntimeError("JPEG augmentation failed")
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result, labels


def write_sample(image_path, label_file, image, labels):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_file.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(image_path.suffix, image)
    if not ok:
        raise RuntimeError("Could not encode augmented image")
    encoded.tofile(str(image_path))
    label_file.write_text("".join(f"{int(c)} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for c, x, y, w, h in labels), encoding="utf-8")


def augment_dataset(config, output, conditions, copies=2, severity=0.4, seed=42):
    if copies < 1 or not conditions or any(c not in CONDITIONS for c in conditions) or not 0 < severity <= 1:
        raise ValueError("Specify valid conditions, positive copies and severity in (0,1]")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Augmentation output already exists; choose a new directory")
    data, _, splits = validate_dataset(config)
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    manifest = {"source_config": str(Path(config).resolve()), "seed": seed, "severity": severity,
                "conditions": list(conditions), "copies_per_train_image": copies, "records": []}
    for split, paths in splits.items():
        for index, source in enumerate(paths):
            stem = f"original_{index:07d}"
            image = output / "images" / split / (stem + source.suffix.lower())
            label = output / "labels" / split / (stem + ".txt")
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, image)
            shutil.copy2(label_path(source), label)
            manifest["records"].append({"split": split, "source": str(source), "image": str(image.relative_to(output)), "condition": "original"})
            if split != "train":
                continue
            original = readable_image(source)
            boxes = read_labels(label_path(source), len(data["names"]))
            for copy in range(copies):
                condition = conditions[int(rng.integers(len(conditions)))]
                sample_seed = int(rng.integers(0, 2**31))
                changed, changed_labels = transform(original, boxes, condition, severity, sample_seed)
                stem = f"aug_{index:07d}_{copy:02d}"
                destination = output / "images" / split / (stem + ".png")
                write_sample(destination, output / "labels" / split / (stem + ".txt"), changed, changed_labels)
                manifest["records"].append({"split": split, "source": str(source), "image": str(destination.relative_to(output)),
                                            "condition": condition, "seed": sample_seed})
    config_out = output / "data.yaml"
    config_out.write_text(yaml.safe_dump({"path": ".", **{s: f"images/{s}" for s in splits}, "names": data["names"]},
                                         allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(output / "augmentation.json", manifest)
    validate_dataset(config_out)
    return config_out


def augment_command(args):
    print(augment_dataset(args.data, args.output, args.conditions, args.copies, args.severity, args.seed))
