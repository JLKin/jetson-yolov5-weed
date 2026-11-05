"""Validate detection labels before invoking upstream training."""
import hashlib
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def class_names(value):
    if isinstance(value, dict):
        converted = {}
        for key, name in value.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)) or not str(key).isdigit():
                raise ValueError("Class IDs must be nonnegative integers")
            category = int(key)
            if category in converted:
                raise ValueError("Duplicate class ID")
            converted[category] = name
        value = converted
        if sorted(value) != list(range(len(value))):
            raise ValueError("Class IDs must start at 0 and be contiguous")
        value = [value[k] for k in range(len(value))]
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
        raise ValueError("names must contain non-empty class names")
    if len(set(value)) != len(value):
        raise ValueError("Class names must be unique")
    return value


def load_dataset(config):
    config = Path(config).resolve()
    data = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must be a mapping")
    root = Path(data.get("path", "."))
    root = (config.parent / root).resolve() if not root.is_absolute() else root.resolve()
    names = class_names(data.get("names"))
    if "nc" in data and int(data["nc"]) != len(names):
        raise ValueError("nc disagrees with names")
    normalized = {"path": str(root), "names": names, "nc": len(names)}
    for split in ("train", "val", "test"):
        values = data.get(split)
        if not values:
            if split != "test":
                raise ValueError(f"Missing {split} split")
            continue
        values = values if isinstance(values, list) else [values]
        normalized[split] = [str((root / Path(v)).resolve()) for v in values]
    return normalized


def image_files(entry):
    entry = Path(entry)
    if entry.is_dir():
        return sorted(p.resolve() for p in entry.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if entry.is_file() and entry.suffix.lower() == ".txt":
        return [(entry.parent / line.strip()).resolve() for line in entry.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
        return [entry.resolve()]
    raise ValueError(f"Dataset entry does not exist or is unsupported: {entry}")


def label_path(image):
    parts = list(Path(image).parts)
    # Match upstream img2label_paths: replace the last /images/ component.
    positions = [i for i, part in enumerate(parts[:-1]) if part == "images"]
    if not positions:
        raise ValueError(f"Image must be below an images directory: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(path, count):
    if not path.is_file():
        raise ValueError(f"Missing label: {path}; backgrounds require an explicit empty .txt")
    labels = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(x) for x in line.split()]
        except ValueError as exc:
            raise ValueError(f"Non-numeric label at {path}:{number}") from exc
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError(f"Expected class cx cy width height at {path}:{number}")
        category, x, y, width, height = values
        if not category.is_integer() or not 0 <= category < count:
            raise ValueError(f"Invalid class ID at {path}:{number}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Coordinates must be normalized to [0,1] at {path}:{number}")
        if min(x - width / 2, y - height / 2) < -1e-4 or max(x + width / 2, y + height / 2) > 1.0001:
            raise ValueError(f"Box extends outside image at {path}:{number}")
        labels.append((int(category), x, y, width, height))
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate boxes in {path}")
    return labels


def validate_dataset(config):
    data = load_dataset(config)
    seen_paths, seen_hashes, seen_labels = {}, {}, {}
    report = {"names": data["names"], "splits": {}}
    resolved = {}
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        files = [p for entry in data[split] for p in image_files(entry)]
        if not files:
            raise ValueError(f"Empty {split} split")
        counts, backgrounds = Counter(), 0
        for path in files:
            if path in seen_paths:
                raise ValueError(f"Repeated image in {seen_paths[path]} and {split}: {path}")
            seen_paths[path] = split
            with Image.open(path) as img:
                if min(img.size) < 10:
                    raise ValueError(f"Image smaller than 10 pixels: {path}")
                img.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes and seen_hashes[digest][0] != split:
                raise ValueError(f"Data leakage across splits: {path} and {seen_hashes[digest][1]}")
            seen_hashes[digest] = (split, str(path))
            label = label_path(path)
            if label in seen_labels:
                raise ValueError(f"Multiple images share one label: {path}, {seen_labels[label]}")
            seen_labels[label] = path
            boxes = read_labels(label, len(data["names"]))
            backgrounds += not boxes
            counts.update(box[0] for box in boxes)
        if not counts and split in {"train", "val"}:
            raise ValueError(f"{split} contains no annotated objects")
        report["splits"][split] = {"images": len(files), "backgrounds": backgrounds,
                                   "objects": sum(counts.values()), "per_class": dict(sorted(counts.items()))}
        resolved[split] = files
    return data, report, resolved


def normalized_config(config, output_dir):
    data, report, resolved = validate_dataset(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, files in resolved.items():
        listing = (output_dir / f"{split}.txt").resolve()
        listing.write_text("\n".join(p.as_posix() for p in files) + "\n", encoding="utf-8")
        data[split] = str(listing)
    target = output_dir / "data.yaml"
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target.resolve(), report


def prepare_dataset(images, labels, output, names, seed=42, group_regex=None):
    """Copy an unsplit dataset; a regex's first capture is the scene/group ID."""
    images, labels, output = map(lambda p: Path(p).resolve(), (images, labels, output))
    names = class_names(names)
    if output.exists():
        raise ValueError("Output directory already exists; choose a new destination")
    files = image_files(images)
    groups, seen = defaultdict(list), set()
    pattern = re.compile(group_regex) if group_regex else None
    for image in files:
        relative = image.relative_to(images)
        label = labels / relative.with_suffix(".txt")
        read_labels(label, len(names))
        if label in seen:
            raise ValueError(f"Multiple image extensions share label: {label}")
        seen.add(label)
        match = pattern.search(relative.as_posix()) if pattern else None
        if pattern and (not match or not match.lastindex):
            raise ValueError(f"Group regex must capture a scene ID for every file: {relative}")
        key = match.group(1) if match else relative.as_posix()
        groups[key].append((image, label, relative))
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least 3 images/groups are needed for train/val/test")
    random.Random(seed).shuffle(keys)
    n_val = max(1, round(len(keys) * 0.1))
    n_test = max(1, round(len(keys) * 0.1))
    n_train = len(keys) - n_val - n_test
    partitions = {"train": keys[:n_train], "val": keys[n_train:n_train + n_val], "test": keys[n_train + n_val:]}
    # Detect duplicate source content before copying anything.
    content_split = {}
    for split, group_keys in partitions.items():
        for key in group_keys:
            for image, _, _ in groups[key]:
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in content_split and content_split[digest] != split:
                    raise ValueError("Duplicate images would cross splits; deduplicate or group them first")
                content_split[digest] = split
    output.mkdir(parents=True)
    for split, group_keys in partitions.items():
        for key in group_keys:
            for image, label, relative in groups[key]:
                for source, destination in [(image, output / "images" / split / relative),
                                            (label, output / "labels" / split / relative.with_suffix(".txt"))]:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    config = output / "data.yaml"
    config.write_text(yaml.safe_dump({"path": ".", **{s: f"images/{s}" for s in partitions}, "names": names},
                                     allow_unicode=True, sort_keys=False), encoding="utf-8")
    validate_dataset(config)
    return config
