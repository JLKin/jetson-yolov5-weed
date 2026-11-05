"""Local media and bounded reconnecting live capture with a one-frame buffer."""
from __future__ import annotations
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .common import utc_now
from .dataset import IMAGE_SUFFIXES


def safe_source(source):
    source = str(source)
    if "://" not in source:
        return source
    try:
        parsed = urlsplit(source)
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{parsed.hostname}{port}/[stream]"
    except ValueError:
        return "[network source]"


@dataclass
class Frame:
    image: object
    index: int
    source: str
    captured_at: str
    media_time_ms: float | None
    fps: float
    is_image: bool = False
    quality: dict | None = None
    preprocessing_ms: float = 0.0


def readable_image(path):
    # imdecode works with Unicode paths on Windows.
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def local_frames(source):
    import cv2
    path = Path(source).resolve()
    if not path.exists():
        raise ValueError(f"Source not found: {path}")
    if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if path.is_dir() else [path]
        if not files:
            raise ValueError("Source directory contains no images")
        for index, file in enumerate(files):
            yield Frame(readable_image(file), index, str(file), utc_now(), None, 1.0, True)
        return
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and 0 < fps <= 240 else 25.0
        index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                if index == 0:
                    raise ValueError(f"No decodable video frames: {path}")
                break
            yield Frame(image, index, str(path), utc_now(), index * 1000 / fps, fps)
            index += 1
    finally:
        cap.release()


class LiveFrames:
    """Capture on a worker thread; the consumer takes the latest available frame."""
    def __init__(self, source, timeout=5.0, reconnects=5, retry_delay=1.0):
        if timeout <= 0 or reconnects < 0 or retry_delay < 0:
            raise ValueError("Timeout must be positive; reconnects and retry delay cannot be negative")
        self.source = int(source) if str(source).isdigit() else str(source)
        self.timeout, self.reconnects, self.retry_delay = timeout, reconnects, retry_delay
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest, self.error, self.done = None, None, False
        self.thread = threading.Thread(target=self._capture, daemon=True, name="video-capture")

    def _open(self):
        import cv2
        if isinstance(self.source, int):
            return cv2.VideoCapture(self.source)
        # Use FFmpeg TCP to reduce packet loss; timeouts also bound shutdown.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.timeout * 1000),
                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.timeout * 1000)])

    def _capture(self):
        import cv2
        failures, index = 0, 0
        cap = None
        try:
            while not self.stop.is_set():
                cap = self._open()
                while cap.isOpened() and not self.stop.is_set():
                    ok, image = cap.read()
                    if not ok:
                        break
                    failures = 0
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fps = fps if math.isfinite(fps) and 0 < fps <= 240 else 25.0
                    with self.condition:
                        self.latest = Frame(image, index, safe_source(self.source), utc_now(), None, fps)
                        index += 1
                        self.condition.notify_all()
                cap.release()
                cap = None
                if self.stop.is_set():
                    break
                failures += 1
                if failures > self.reconnects:
                    raise RuntimeError("Live source unavailable after consecutive reconnect attempts")
                logging.warning("Live source disconnected; retry %d/%d", failures, self.reconnects)
                self.stop.wait(self.retry_delay)
        except Exception:
            # Native decoder exceptions can contain credentials; do not propagate them.
            self.error = RuntimeError("Live capture failed. Check camera connectivity, URL, codec and reconnect settings.")
        finally:
            if cap is not None:
                cap.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def __iter__(self):
        self.thread.start()
        last = -1
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.done or (self.latest is not None and self.latest.index != last), timeout=0.5)
                    frame = self.latest
                    if frame is not None and frame.index != last:
                        last = frame.index
                    elif self.done:
                        if self.error:
                            raise self.error
                        return
                    else:
                        continue
                yield frame
        finally:
            self.stop.set()
            self.thread.join(timeout=self.timeout + 1)


def frames(source, timeout=5.0, reconnects=5):
    text = str(source)
    if text.isdigit() or text.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return iter(LiveFrames(source, timeout, reconnects))
    return local_frames(source)
