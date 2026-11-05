import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from weed_detector.augmentation import CONDITIONS, augment_dataset, transform
from weed_detector.benchmarks import accuracy_decision, latency_stats
from weed_detector.dataset import validate_dataset
from weed_detector.vision import FrameProcessor, flip_labels, quality_metrics, restore_detections


class VisionTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((96, 128))
        gray = (((x // 4 + y // 4) % 2) * 200 + 20).astype(np.uint8)
        self.image = np.repeat(gray[:, :, None], 3, axis=2)
        self.labels = [(0, .2, .3, .2, .2)]

    def test_flip_boxes_in_both_coordinate_systems(self):
        labels = flip_labels(self.labels, True, True)
        np.testing.assert_allclose(labels[0][1:], [.8, .7, .2, .2])
        detection = {"bbox_xyxy": [10, 20, 30, 50], "class_id": 0}
        once = restore_detections([detection], 128, 96, "both")
        self.assertEqual(once[0]["bbox_xyxy"], [98, 46, 118, 76])
        self.assertEqual(restore_detections(once, 128, 96, "both"), [detection])

    def test_mirror_matches_image_and_preserves_source(self):
        original = self.image.copy()
        changed, labels = transform(self.image, self.labels, "mirror-horizontal")
        np.testing.assert_array_equal(changed, original[:, ::-1])
        self.assertAlmostEqual(labels[0][1], .8)
        np.testing.assert_array_equal(original, self.image)

    def test_quality_blur_rank_and_highlight_flags(self):
        blurred = cv2.GaussianBlur(self.image, (0, 0), 3)
        self.assertLess(quality_metrics(blurred)["sharpness_laplacian"], quality_metrics(self.image)["sharpness_laplacian"])
        self.assertIn("highlight_risk", quality_metrics(np.full_like(self.image, 255))["flags"])
        self.assertIn("dark_risk", quality_metrics(np.zeros_like(self.image))["flags"])

    def test_default_preprocessing_does_not_change_pixels(self):
        changed, report, elapsed = FrameProcessor().apply(self.image)
        np.testing.assert_array_equal(changed, self.image)
        self.assertEqual(report["enhance"], "none")
        self.assertGreaterEqual(elapsed, 0)

    def test_enhancement_does_not_mutate_input_and_gamma_direction(self):
        original = self.image.copy()
        brightened, _, _ = FrameProcessor(enhance="gamma", gamma=.5).apply(self.image)
        self.assertGreater(float(brightened.mean()), float(original.mean()))
        corrected, _, _ = FrameProcessor(enhance="clahe").apply(self.image)
        self.assertEqual(corrected.shape, original.shape)
        np.testing.assert_array_equal(self.image, original)

    def test_transform_shapes_labels_and_reproducibility(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                first, labels = transform(self.image, self.labels, condition, .5, 4)
                second, _ = transform(self.image, self.labels, condition, .5, 4)
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first.dtype, np.uint8)
                self.assertEqual(first.shape, self.image.shape)
                self.assertEqual(len(labels), 1)
                if not condition.startswith("mirror"):
                    self.assertEqual(labels, self.labels)

    def test_invalid_configuration(self):
        for arguments in [{"gamma": 0}, {"gamma": float("nan")}, {"highlight_threshold": 2}, {"mirror": "unknown"}]:
            with self.assertRaises(ValueError):
                FrameProcessor(**arguments)
        with self.assertRaises(ValueError):
            transform(self.image, self.labels, "glare", 0)

    def test_latency_and_accuracy_gate(self):
        stats = latency_stats([10, 20, 30])
        self.assertEqual(stats["mean_ms"], 20)
        self.assertEqual(stats["pipeline_fps"], 50)
        base = {"mAP50_95": .8, "recall": .9}
        good = {"mAP50_95": .79, "recall": .88}
        bad = {"mAP50_95": .75, "recall": .88}
        self.assertTrue(accuracy_decision(base, good, .01, .02)["passed"])
        self.assertFalse(accuracy_decision(base, bad, .01, .02)["passed"])
        with self.assertRaises(ValueError):
            latency_stats([float("nan")])


class AugmentationIsolationTests(unittest.TestCase):
    def test_only_train_augmented_and_heldout_bytes_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            originals = {}
            for i, split in enumerate(("train", "val", "test")):
                image_path = raw / "images" / split / "source.png"
                label = raw / "labels" / split / "source.txt"
                image_path.parent.mkdir(parents=True)
                label.parent.mkdir(parents=True)
                cv2.imencode(".png", np.full((40, 60, 3), 50 + i, np.uint8))[1].tofile(str(image_path))
                label.write_text("0 .2 .3 .2 .2\n", encoding="utf-8")
                originals[split] = hashlib.sha256(image_path.read_bytes()).hexdigest()
            config = raw / "data.yaml"
            config.write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "names": ["weed"]}))
            result = augment_dataset(config, root / "augmented", ["mirror-horizontal"], copies=2)
            _, report, files = validate_dataset(result)
            self.assertEqual([report["splits"][s]["images"] for s in ("train", "val", "test")], [3, 1, 1])
            for split in ("val", "test"):
                self.assertEqual(hashlib.sha256(files[split][0].read_bytes()).hexdigest(), originals[split])
            manifest = json.loads((result.parent / "augmentation.json").read_text())
            self.assertTrue(all(r["condition"] == "original" for r in manifest["records"] if r["split"] != "train"))


if __name__ == "__main__":
    unittest.main()
