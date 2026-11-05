import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from weed_detector.common import new_run
from weed_detector.results import ResultWriter
from weed_detector.sources import Frame, LiveFrames, local_frames, safe_source


class OutputTests(unittest.TestCase):
    def test_zero_detection_frame_is_preserved_and_box_coordinates_written(self):
        with tempfile.TemporaryDirectory() as temp:
            frame = Frame(np.zeros((100, 200, 3), dtype=np.uint8), 0, "example.mp4", "2026-11-05T00:00:00Z", 0, 25)
            writer = ResultWriter(temp, visual=False)
            writer.write(frame, [], 2)
            frame.index = 1
            frame.quality = {"flags": ["blur_risk", "highlight_risk"]}
            frame.preprocessing_ms = 1.5
            writer.write(frame, [{"class_id": 0, "class_name": "杂草", "confidence": .9, "bbox_xyxy": [20, 10, 100, 80]}], 4)
            writer.close("failed", "test failure")
            writer.close()
            records = [json.loads(line) for line in (Path(temp) / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["detection_count"] for r in records], [0, 1])
            with (Path(temp) / "detections.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["class_name"], "杂草")
            self.assertEqual((rows[0]["x2"], rows[0]["width"]), ("100", "200"))
            summary = json.loads((Path(temp) / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["mean_inference_ms"], 3)
            self.assertEqual(records[1]["schema_version"], 2)
            self.assertEqual(summary["quality_flag_frames"], {"blur_risk": 1, "highlight_risk": 1})
            self.assertIn("blur_risk", rows[0]["quality_flags"])

    def test_run_paths_cannot_escape_project_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = new_run(temp, "weed"), new_run(temp, "weed")
            self.assertNotEqual(first, second)
            with self.assertRaises(ValueError):
                new_run(temp, "../outside")

    def test_source_redaction(self):
        source = safe_source("rtsp://admin:secret@192.168.1.10:554/live?token=secret")
        self.assertNotIn("secret", source)
        self.assertNotIn("admin", source)
        self.assertIn("192.168.1.10:554", source)

    def test_unicode_image_and_exhaustion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "杂草.png"
            Image.new("RGB", (40, 30), "green").save(path)
            frames = list(local_frames(path))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].image.shape, (30, 40, 3))

    def test_network_retries_are_bounded(self):
        class ClosedCapture:
            def isOpened(self):
                return False

            def release(self):
                pass

        source = LiveFrames("rtsp://invalid/stream", timeout=.1, reconnects=2, retry_delay=0)
        with patch.object(source, "_open", return_value=ClosedCapture()) as opened:
            with self.assertRaisesRegex(RuntimeError, "Live capture failed"):
                list(source)
            self.assertEqual(opened.call_count, 3)
        self.assertFalse(source.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
