"""Run original YOLOv5 with shared preprocessing and postprocessing."""
import time
from pathlib import Path

import numpy as np

from .common import import_upstream


class Detector:
    def __init__(self, weights, device="0", imgsz=640, confidence=0.25, iou=0.45,
                 half=False, data=None, max_det=300):
        if not Path(weights).is_file():
            raise ValueError(f"Model not found: {weights}")
        if Path(weights).suffix not in {".pt", ".engine"}:
            raise ValueError("Inference supports .pt and .engine; ONNX is an export intermediate")
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1 or imgsz < 32 or max_det < 1:
            raise ValueError("Invalid confidence, IoU, image size or maximum detections")
        import_upstream()
        import torch
        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        from utils.torch_utils import select_device

        self.torch = torch
        self.device = select_device(device)
        if Path(weights).suffix == ".engine":
            if self.device.type != "cuda":
                raise ValueError("TensorRT inference requires --device 0")
            if not data or not Path(data).is_file():
                raise ValueError("TensorRT inference requires --data with the trained class names")
            import tensorrt as trt
            if not 8 <= int(trt.__version__.split(".")[0]) <= 10:
                raise ValueError("This backend supports TensorRT 8-10")
        self.model = DetectMultiBackend(str(Path(weights).resolve()), device=self.device, fp16=half,
                                        data=str(Path(data).resolve()) if data else None)
        if self.model.engine and (not data or not Path(data).is_file()):
            raise ValueError("TensorRT inference requires --data with the trained class names")
        self.imgsz = check_img_size(imgsz, s=self.model.stride)
        self.confidence, self.iou, self.max_det = confidence, iou, max_det
        self.names = self.model.names
        # Exported engines are batch 1 with square, static input shape.
        if self.model.engine:
            shape = tuple(self.model.bindings["images"].shape)
            if shape != (1, 3, self.imgsz, self.imgsz):
                raise ValueError(f"Engine shape {shape} disagrees with --imgsz {self.imgsz}")
            output = self.model.bindings[self.model.output_names[0]].shape
            if output[-1] != len(self.names) + 5:
                raise ValueError("Class names do not match the engine output")
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))

    def predict(self, bgr):
        from utils.dataloaders import letterbox
        from utils.general import non_max_suppression, scale_boxes

        torch = self.torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        image = letterbox(bgr, self.imgsz, stride=self.model.stride, auto=False)[0]
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[::-1])
        image = torch.from_numpy(image).to(self.device)
        image = (image.half() if self.model.fp16 else image.float()) / 255.0
        with torch.inference_mode():
            predictions = self.model(image.unsqueeze(0))
            predictions = non_max_suppression(predictions, self.confidence, self.iou, max_det=self.max_det)
            boxes = predictions[0]
            boxes[:, :4] = scale_boxes(image.shape[1:], boxes[:, :4], bgr.shape).round()
            rows = boxes.cpu().tolist()
        detections = []
        for x1, y1, x2, y2, score, category in rows:
            category = int(category)
            detections.append({"class_id": category, "class_name": self.names[category],
                               "confidence": round(score, 6), "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)]})
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return detections, (time.perf_counter() - started) * 1000
