import shutil
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from weed_detector.dataset import class_names, normalized_config, prepare_dataset, read_labels, validate_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for i, split in enumerate(("train", "val", "test")):
            images = self.root / "images" / split
            labels = self.root / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (100, 80), (30 + i, 20, 40)).save(images / "a.png")
            (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        self.config = self.root / "data.yaml"
        self.config.write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                                               "test": "images/test", "names": ["weed"]}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_and_absolute_normalized_paths(self):
        _, report, _ = validate_dataset(self.config)
        self.assertEqual(report["splits"]["train"]["objects"], 1)
        config, _ = normalized_config(self.config, self.root / "normalized")
        data = yaml.safe_load(config.read_text())
        self.assertTrue(Path(data["train"]).is_absolute())
        self.assertEqual(Path(Path(data["train"]).read_text().strip()), self.root / "images" / "train" / "a.png")

    def test_missing_label_is_not_silent_background(self):
        (self.root / "labels" / "train" / "a.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing label"):
            validate_dataset(self.config)

    def test_explicit_empty_background(self):
        Image.new("RGB", (100, 80), (90, 90, 90)).save(self.root / "images" / "train" / "background.png")
        (self.root / "labels" / "train" / "background.txt").write_text("")
        _, report, _ = validate_dataset(self.config)
        self.assertEqual(report["splits"]["train"]["backgrounds"], 1)

    def test_cross_split_content_leakage(self):
        shutil.copy2(self.root / "images" / "train" / "a.png", self.root / "images" / "test" / "a.png")
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_dataset(self.config)

    def test_bad_labels(self):
        path = self.root / "bad.txt"
        for content in ["2 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .5 .5 0 .2", "0 .01 .5 .2 .2",
                        "0.5 .5 .5 .2 .2", "0 .1 .1 .2 .2 .3 .3", "0 .5 .5 .2 .2\n0 .5 .5 .2 .2"]:
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_labels(path, 1)

    def test_empty_validation_objects(self):
        (self.root / "labels" / "val" / "a.txt").write_text("")
        with self.assertRaisesRegex(ValueError, "no annotated"):
            validate_dataset(self.config)

    def test_class_ids_and_names(self):
        self.assertEqual(class_names({1: "crop", 0: "weed"}), ["weed", "crop"])
        for value in [{1: "weed"}, {0.5: "weed"}, {True: "weed"}, ["weed", "weed"], [], [""]]:
            with self.assertRaises(ValueError):
                class_names(value)

    def test_scene_group_split(self):
        images, labels = self.root / "raw", self.root / "raw-labels"
        images.mkdir()
        labels.mkdir()
        for scene in range(5):
            for frame in range(2):
                stem = f"scene{scene}_frame{frame}"
                Image.new("RGB", (40, 40), (10 + scene, 10 + frame, 30)).save(images / f"{stem}.png")
                (labels / f"{stem}.txt").write_text("0 .5 .5 .2 .2")
        prepared = prepare_dataset(images, labels, self.root / "prepared", ["weed"], group_regex=r"(scene\d+)_")
        _, _, files = validate_dataset(prepared)
        locations = {}
        for split, paths in files.items():
            for path in paths:
                scene = path.stem.split("_")[0]
                self.assertIn(locations.get(scene, split), [split])
                locations[scene] = split

    def test_prepare_does_not_overwrite(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_dataset(self.root / "images", self.root / "labels", self.root, ["weed"])


if __name__ == "__main__":
    unittest.main()
