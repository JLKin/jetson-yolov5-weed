"""Durable per-frame records, visualizations and end-of-run summary."""
import csv
import json
import time
from collections import Counter
from pathlib import Path

from .common import utc_now, write_json

CSV_COLUMNS = ["frame_index", "captured_at", "media_time_ms", "source", "width", "height",
               "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "inference_ms",
               "preprocessing_ms", "quality_flags"]


class ResultWriter:
    def __init__(self, directory, visual=True, segment_seconds=60, output_fps=None):
        if segment_seconds <= 0 or (output_fps is not None and not 0 < output_fps <= 240):
            raise ValueError("Segment duration must be positive and output FPS in (0,240]")
        self.directory = Path(directory)
        self.visual, self.segment_seconds, self.output_fps = visual, segment_seconds, output_fps
        self.jsonl = (self.directory / "detections.jsonl").open("w", encoding="utf-8")
        self.csv_file = (self.directory / "detections.csv").open("w", encoding="utf-8-sig", newline="")
        self.csv = csv.DictWriter(self.csv_file, fieldnames=CSV_COLUMNS)
        self.csv.writeheader()
        self.video, self.video_shape, self.video_frames, self.segment = None, None, 0, 0
        self.started, self.started_at = time.perf_counter(), utc_now()
        self.frames, self.objects, self.ms = 0, Counter(), 0.0
        self.quality_flags, self.preprocessing_ms = Counter(), 0.0
        self.closed = False

    def write(self, frame, detections, inference_ms):
        height, width = frame.image.shape[:2]
        base = {"frame_index": frame.index, "captured_at": frame.captured_at,
                "media_time_ms": frame.media_time_ms, "source": frame.source,
                "width": width, "height": height, "inference_ms": round(inference_ms, 3),
                "preprocessing_ms": round(frame.preprocessing_ms, 3),
                "quality_flags": "|".join((frame.quality or {}).get("flags", []))}
        record = {"schema_version": 2, **base, "quality": frame.quality, "detection_count": len(detections), "detections": detections}
        self.jsonl.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            self.csv.writerow({**base, **{key: detection[key] for key in ("class_id", "class_name", "confidence")},
                               "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            self.objects[detection["class_name"]] += 1
        self.jsonl.flush()
        self.csv_file.flush()
        self.frames += 1
        self.ms += inference_ms
        self.preprocessing_ms += frame.preprocessing_ms
        self.quality_flags.update((frame.quality or {}).get("flags", []))
        if self.visual:
            self._visual(frame, detections)

    def _visual(self, frame, detections):
        import cv2
        canvas = frame.image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)
            # ASCII class ID remains legible even with Chinese dataset class names.
            label = f"#{detection['class_id']} {detection['confidence']:.2f}"
            cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (60, 220, 60), 2)
        if frame.is_image:
            destination = self.directory / f"frame_{self.frames:06d}_{Path(frame.source).stem}.jpg"
            ok, encoded = cv2.imencode(".jpg", canvas)
            if not ok:
                raise RuntimeError("Could not encode annotated image")
            encoded.tofile(str(destination))
            return
        shape = (canvas.shape[1], canvas.shape[0])
        fps = self.output_fps or frame.fps
        limit = max(1, round(fps * self.segment_seconds))
        if self.video is None or self.video_shape != shape or self.video_frames >= limit:
            if self.video is not None:
                self.video.release()
            self.segment += 1
            self.video = cv2.VideoWriter(str(self.directory / f"annotated_{self.segment:04d}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, shape)
            if not self.video.isOpened():
                raise RuntimeError("Cannot open MP4 writer; install a codec-enabled OpenCV or use --no-visual")
            self.video_shape, self.video_frames = shape, 0
        self.video.write(canvas)
        self.video_frames += 1

    def close(self, status="completed", error=None):
        if self.closed:
            return
        self.closed = True
        self.jsonl.close()
        self.csv_file.close()
        if self.video is not None:
            self.video.release()
        elapsed = time.perf_counter() - self.started
        summary = {"status": status, "started_at": self.started_at, "finished_at": utc_now(),
                   "processed_frames": self.frames, "detections_by_class": dict(self.objects),
                   "count_note": "Per-frame detections; not unique plants or tracked objects.",
                   "elapsed_seconds": round(elapsed, 3),
                   "processed_fps": round(self.frames / elapsed, 3) if elapsed else 0,
                   "mean_inference_ms": round(self.ms / self.frames, 3) if self.frames else None,
                   "mean_preprocessing_ms": round(self.preprocessing_ms / self.frames, 3) if self.frames else None,
                   "quality_flag_frames": dict(self.quality_flags)}
        if error:
            summary["error"] = error
        write_json(self.directory / "summary.json", summary)
