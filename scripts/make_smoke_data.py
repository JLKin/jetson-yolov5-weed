"""Create synthetic engineering-test fixtures. These are NOT weed training data."""
import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def create(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Smoke data directory already exists")
    rng = random.Random(17)
    for split, count in [("train", 12), ("val", 4), ("test", 4)]:
        images, labels = output / "images" / split, output / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(count):
            image = Image.new("RGB", (320, 240), (rng.randrange(15, 50), rng.randrange(15, 50), rng.randrange(15, 50)))
            draw = ImageDraw.Draw(image)
            x, y = rng.randrange(20, 220), rng.randrange(20, 140)
            w, h = rng.randrange(30, 80), rng.randrange(30, 80)
            draw.ellipse((x, y, x + w, y + h), fill=(rng.randrange(20, 80), rng.randrange(130, 250), 50))
            draw.text((5, 5), f"SYNTHETIC {split} {index}", fill="white")
            image.save(images / f"fixture_{index:03d}.jpg")
            (labels / f"fixture_{index:03d}.txt").write_text(
                f"0 {(x + w / 2) / 320:.6f} {(y + h / 2) / 240:.6f} {w / 320:.6f} {h / 240:.6f}\n", encoding="utf-8")
    config = output / "data.yaml"
    config.write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                                      "test": "images/test", "names": ["synthetic_test_target"]}), encoding="utf-8")
    import cv2
    import numpy as np
    video = cv2.VideoWriter(str(output / "test.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    if not video.isOpened():
        raise RuntimeError("Smoke video encoder unavailable")
    try:
        for index in range(20):
            canvas = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.circle(canvas, (40 + index * 8, 120), 25, (30, 200, 30), -1)
            video.write(canvas)
    finally:
        video.release()
    print(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="work/smoke-data")
    create(parser.parse_args().output)
