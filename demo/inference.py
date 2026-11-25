"""Model loading, inference and box drawing for the vehicle detection demo. No UI code here."""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Index = YOLO class id, as in data/main/data.yaml and the YOLOv8 checkpoint's `names`.
# The torchvision models (SSD, Faster R-CNN) were trained with label = index + 1 (0 = background).
CLASS_NAMES = ("Bus", "Car", "Motorbike", "Truck")
# RGB; categorical palette validated for colour-vision deficiency. Text labels carry identity too.
COLORS = {"Car": (0x25, 0x7D, 0xE0), "Motorbike": (0xF4, 0x72, 0x30), "Truck": (0x00, 0xB8, 0x94), "Bus": (0xF2, 0xA6, 0x00)}

MIN_SIDE, MAX_SIDE = 32, 4096
YOLO_IMGSZ = 832  # train_args.imgsz stored in models/yolov8.pt
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255  # albumentations A.Normalize() defaults
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


@dataclass(frozen=True)
class ModelSpec:
    file: str
    preprocessing: str
    default_conf: float
    supports_iou: bool
    input_size: int


MODEL_SPECS = {
    "YOLOv8n": ModelSpec("yolov8.pt", f"Letterbox to {YOLO_IMGSZ}x{YOLO_IMGSZ} (aspect ratio preserved)", 0.25, True, YOLO_IMGSZ),
    "Faster R-CNN": ModelSpec("best_FasterRCNN.pth", "Resize to 640x640 + ImageNet normalization (training val_transform)", 0.5, False, 640),
    "SSD300": ModelSpec("best_SSD.pth", "Resize to 300x300 + ImageNet normalization (training val_transform)", 0.5, False, 300),
}
DEFAULT_MODEL = "YOLOv8n"


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in original-image pixels


@dataclass(frozen=True)
class Result:
    detections: list
    inference_ms: float  # model forward pass only
    total_ms: float  # preprocessing + inference + postprocessing
    model: str
    device: str


def _checkpoint(spec):
    path = MODELS_DIR / spec.file
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}. See the 'Live demo' section of the project README.")
    return path


def _load_yolo(spec, warmup=True):
    from ultralytics import YOLO

    model = YOLO(str(_checkpoint(spec)))
    names = {int(k): v.lower() for k, v in model.names.items()}
    # Refuse a generic (e.g. COCO) checkpoint silently standing in for the project model.
    if names != {i: c.lower() for i, c in enumerate(CLASS_NAMES)}:
        raise RuntimeError(f"Unexpected YOLO class mapping {names}; expected {CLASS_NAMES}.")
    # Ultralytics builds its predictor on the first predict(); do it here so it is not billed to a user request.
    model.to(DEVICE)
    if warmup:
        model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=YOLO_IMGSZ, device=DEVICE, verbose=False)
    return model


def _load_torchvision(builder, spec, warmup=True):
    model = builder(_checkpoint(spec))
    model.eval().to(DEVICE)
    if warmup:
        with torch.inference_mode():  # ZeroGPU must defer computation until a decorated call.
            model([torch.zeros(3, spec.input_size, spec.input_size, device=DEVICE)])
    return model


def _build_faster_rcnn(path):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, len(CLASS_NAMES) + 1)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def _build_ssd(path):
    from torchvision.models.detection import ssd300_vgg16
    from torchvision.models.detection.ssd import SSDClassificationHead

    model = ssd300_vgg16(weights=None, weights_backbone=None)
    model.head.classification_head = SSDClassificationHead(
        [m.in_channels for m in model.head.classification_head.module_list],
        model.anchor_generator.num_anchors_per_location(),
        len(CLASS_NAMES) + 1,
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


_LOADERS = {
    "YOLOv8n": _load_yolo,
    "Faster R-CNN": lambda spec, warmup=True: _load_torchvision(_build_faster_rcnn, spec, warmup),
    "SSD300": lambda spec, warmup=True: _load_torchvision(_build_ssd, spec, warmup),
}
_models = {}
_lock = threading.Lock()


def get_model(name, *, warmup=True):
    """Load a model once per process and reuse it for every request."""
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{name}'. Choose one of {list(MODEL_SPECS)}.")
    with _lock:
        if name not in _models:
            _models[name] = _LOADERS[name](MODEL_SPECS[name], warmup=warmup)
        return _models[name]


def validate_image(image):
    """Return a contiguous RGB uint8 array, or raise ValueError with a user-facing message."""
    if image is None:
        raise ValueError("No image provided. Upload an image or pick an example.")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Unsupported image format. Use a JPEG or PNG photo.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError("Unsupported number of colour channels.")
    h, w = image.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise ValueError(f"Image is {w}x{h}px; each side must be between {MIN_SIDE} and {MAX_SIDE}px.")
    return np.ascontiguousarray(image)


def _run_yolo(model, image, conf, iou):
    r = model.predict(np.ascontiguousarray(image[:, :, ::-1]), imgsz=YOLO_IMGSZ, conf=conf, iou=iou, device=DEVICE, verbose=False)[0]
    boxes = r.boxes
    rows = zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy())
    return [(tuple(b), int(c), float(s)) for b, c, s in rows], r.speed["inference"]


def _run_torchvision(model, image, conf, size):
    h, w = image.shape[:2]
    x = (cv2.resize(image, (size, size)).astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).to(DEVICE)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model([x])[0]  # torchvision applies its own NMS internally
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - start) * 1000
    scale = np.array([w / size, h / size, w / size, h / size], dtype=np.float32)
    keep = out["scores"] >= conf
    rows = zip(out["boxes"][keep].cpu().numpy() * scale, out["labels"][keep].cpu().numpy() - 1, out["scores"][keep].cpu().numpy())
    return [(tuple(b), int(c), float(s)) for b, c, s in rows], inference_ms


def detect(image, model_name=DEFAULT_MODEL, conf=None, iou=0.7, classes=None):
    """Run one model on an RGB image. `classes` restricts output to these class names (None = all)."""
    image = validate_image(image)
    spec = MODEL_SPECS.get(model_name)
    model = get_model(model_name)  # loading time is excluded from the reported latency
    conf = spec.default_conf if conf is None else float(conf)
    if not 0 < conf <= 1 or not 0 < iou <= 1:
        raise ValueError("Thresholds must be in (0, 1].")

    start = time.perf_counter()
    if spec.supports_iou:
        raw, inference_ms = _run_yolo(model, image, conf, iou)
    else:
        raw, inference_ms = _run_torchvision(model, image, conf, spec.input_size)
    h, w = image.shape[:2]
    wanted = set(CLASS_NAMES if classes is None else classes)
    detections = sorted(
        (
            Detection(CLASS_NAMES[c], s, (max(0.0, float(b[0])), max(0.0, float(b[1])), min(float(w), float(b[2])), min(float(h), float(b[3]))))
            for b, c, s in raw
            if 0 <= c < len(CLASS_NAMES) and CLASS_NAMES[c] in wanted
        ),
        key=lambda d: -d.confidence,
    )
    total_ms = (time.perf_counter() - start) * 1000
    return Result(detections, inference_ms, total_ms, model_name, DEVICE)


def count_by_class(detections):
    counts = {name: 0 for name in CLASS_NAMES}
    for d in detections:
        counts[d.class_name] += 1
    return counts


def text_color(rgb):
    """Black or white, whichever reads better on the given background colour."""
    return (0, 0, 0) if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 140 else (255, 255, 255)


def draw(image, detections):
    """Return a copy of the RGB image with class-coloured boxes and 'Class 0.87' labels."""
    out = image.copy()
    short = min(out.shape[:2])
    thickness = max(1, round(short / 320))
    font_scale = max(0.4, short / 1300)
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        color = COLORS[d.class_name]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = y1 - th - base - 2 if y1 - th - base - 2 >= 0 else y1  # flip inside the box at the top edge
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + base + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color(color), 1, cv2.LINE_AA)
    return out
