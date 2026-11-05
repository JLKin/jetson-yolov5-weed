"""Evaluate fixed corruption cases on held-out data with synchronized labels."""
from pathlib import Path

import yaml

from .augmentation import transform, write_sample
from .common import new_run, write_json
from .dataset import label_path, read_labels, validate_dataset
from .sources import readable_image


def stress_test(args):
    from .evaluation import pipeline_metrics
    from .augmentation import CONDITIONS
    if not 0 < args.severity <= 1 or any(c not in CONDITIONS for c in args.conditions):
        raise ValueError("Invalid stress condition or severity")
    data, _, splits = validate_dataset(args.data)
    if args.split not in splits:
        raise ValueError(f"Dataset has no {args.split} split")
    directory = new_run(args.project, args.name)
    cases = ["baseline", *dict.fromkeys(args.conditions)]
    report = {"source_config": str(Path(args.data).resolve()), "split": args.split,
              "severity": args.severity, "seed": args.seed, "cases": {},
              "scope": "Synthetic diagnostic conditions, not a substitute for real glare/blur field test sets."}
    for condition in cases:
        config = Path(args.data).resolve()
        if condition != "baseline":
            case_root = directory / "datasets" / condition
            case_root.mkdir(parents=True)
            candidate = {"path": str(case_root.resolve()), "names": data["names"]}
            for split, files in splits.items():
                if split == args.split:
                    for index, source in enumerate(files):
                        image, boxes = transform(readable_image(source), read_labels(label_path(source), len(data["names"])),
                                                 condition, args.severity, args.seed + index)
                        stem = f"stress_{index:07d}"
                        write_sample(case_root / "images" / split / (stem + ".png"),
                                     case_root / "labels" / split / (stem + ".txt"), image, boxes)
                    candidate[split] = str((case_root / "images" / split).resolve())
                else:
                    listing = case_root / f"{split}.txt"
                    listing.write_text("\n".join(p.as_posix() for p in files) + "\n", encoding="utf-8")
                    candidate[split] = str(listing.resolve())
            config = case_root / "data.yaml"
            config.write_text(yaml.safe_dump(candidate, allow_unicode=True), encoding="utf-8")
        metrics = pipeline_metrics(args.weights, str(config), args.split, args.device, args.imgsz,
                                   directory / "evaluations" / condition)
        report["cases"][condition] = metrics
        if condition != "baseline":
            metrics["mAP50_95_drop_vs_baseline"] = report["cases"]["baseline"]["mAP50_95"] - metrics["mAP50_95"]
        write_json(directory / "stress-report.json", report)
    print(f"Stress report: {directory / 'stress-report.json'}")
