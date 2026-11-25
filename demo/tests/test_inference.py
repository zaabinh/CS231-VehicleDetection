"""Tests run real inference with the project's checkpoints; nothing is mocked."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference  # noqa: E402
from inference import CLASS_NAMES, COLORS, MODEL_SPECS, Detection, count_by_class, detect, draw, get_model, validate_image  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name):
    return cv2.cvtColor(cv2.imread(str(EXAMPLES / name)), cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def dense():
    # Test-split image with ground truth: 4 cars, 17 motorbikes.
    return load("dense_motorbikes.jpg")


@pytest.mark.parametrize("spec", MODEL_SPECS.values(), ids=list(MODEL_SPECS))
def test_model_artifact_is_real_weights_not_lfs_pointer(spec):
    path = inference.MODELS_DIR / spec.file
    assert path.is_file(), f"missing {path}"
    assert path.stat().st_size > 1_000_000
    assert not path.read_bytes()[:40].startswith(b"version https://git-lfs")


def test_yolo_loads_once_with_project_class_mapping():
    model = get_model("YOLOv8n")
    assert get_model("YOLOv8n") is model
    assert model.names == {0: "bus", 1: "car", 2: "motorbike", 3: "truck"}
    assert CLASS_NAMES == ("Bus", "Car", "Motorbike", "Truck")


def test_yolo_can_preload_without_zero_gpu_warmup(monkeypatch):
    monkeypatch.setattr(inference, "_models", {})
    model = get_model("YOLOv8n", warmup=False)
    assert model.predictor is None
    assert get_model("YOLOv8n", warmup=False) is model


def test_unknown_model_rejected(dense):
    with pytest.raises(ValueError):
        detect(dense, "yolov99")


@pytest.mark.parametrize(
    "bad",
    [None, "not an image", np.zeros((64, 64, 3), np.float32), np.zeros((10, 10, 3), np.uint8),
     np.zeros((4097, 64, 3), np.uint8), np.zeros((64, 64, 5), np.uint8)],
    ids=["empty", "string", "float", "too-small", "too-large", "5-channel"],
)
def test_invalid_input_raises_value_error(bad):
    with pytest.raises(ValueError):
        validate_image(bad)


def test_grayscale_and_rgba_are_converted_to_rgb(dense):
    assert validate_image(cv2.cvtColor(dense, cv2.COLOR_RGB2GRAY)).shape == dense.shape
    assert validate_image(cv2.cvtColor(dense, cv2.COLOR_RGB2RGBA)).shape == dense.shape


def test_yolo_real_inference_matches_ground_truth(dense):
    result = detect(dense, "YOLOv8n", conf=0.25)
    h, w = dense.shape[:2]
    assert result.inference_ms > 0 and result.total_ms >= result.inference_ms
    for d in result.detections:
        assert isinstance(d, Detection) and d.class_name in CLASS_NAMES and 0.25 <= d.confidence <= 1
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    assert [d.confidence for d in result.detections] == sorted((d.confidence for d in result.detections), reverse=True)
    counts = count_by_class(result.detections)
    assert sum(counts.values()) == len(result.detections)
    assert counts["Car"] == 4 and 13 <= counts["Motorbike"] <= 19  # ground truth 4 / 17


def test_higher_confidence_gives_subset(dense):
    low = detect(dense, "YOLOv8n", conf=0.25).detections
    high = detect(dense, "YOLOv8n", conf=0.8).detections
    assert len(high) < len(low)
    assert all(d.confidence >= 0.8 for d in high)
    assert {d.box for d in high} <= {d.box for d in low}


def test_class_filter(dense):
    full = detect(dense, "YOLOv8n").detections
    cars = detect(dense, "YOLOv8n", classes=["Car"]).detections
    assert cars and all(d.class_name == "Car" for d in cars)
    assert len(cars) == count_by_class(full)["Car"]


def test_no_detections_on_blank_image():
    blank = np.full((480, 640, 3), 128, np.uint8)
    result = detect(blank, "YOLOv8n")
    assert result.detections == []
    assert count_by_class(result.detections) == {c: 0 for c in CLASS_NAMES}
    assert np.array_equal(draw(blank, result.detections), blank)


def test_count_by_class():
    dets = [Detection("Car", 0.9, (0, 0, 1, 1)), Detection("Car", 0.8, (0, 0, 1, 1)), Detection("Bus", 0.7, (0, 0, 1, 1))]
    assert count_by_class(dets) == {"Bus": 1, "Car": 2, "Motorbike": 0, "Truck": 0}


def test_annotation_draws_each_box_in_its_class_colour(dense):
    dets = detect(dense, "YOLOv8n").detections
    out = draw(dense, dets)
    assert out.shape == dense.shape and not np.array_equal(out, dense)
    # Bottom-edge midpoint of each box carries its class colour (a few may be covered by overlapping boxes/labels).
    hits = sum(tuple(out[min(round(d.box[3]), out.shape[0] - 1), round((d.box[0] + d.box[2]) / 2)]) == COLORS[d.class_name] for d in dets)
    assert hits >= 0.8 * len(dets)


@pytest.mark.parametrize("name", ["Faster R-CNN", "SSD300"])
def test_torchvision_models_real_inference(dense, name):
    result = detect(dense, name)
    assert result.detections, f"{name} found nothing"
    assert all(d.class_name in CLASS_NAMES and d.confidence >= MODEL_SPECS[name].default_conf for d in result.detections)
    assert count_by_class(result.detections)["Car"] >= 3  # ground truth 4


def test_app_handler_end_to_end(dense):
    import gradio as gr

    import app

    image, summary, table = app.run_detection(dense, "YOLOv8n", 0.25, 0.7, ["Car", "Motorbike", "Truck", "Bus"])
    assert image.shape == dense.shape
    assert f"| **{len(table)}** |" in summary and "ms" in summary
    assert list(table.columns) == ["ID", "Class", "Confidence", "x1", "y1", "x2", "y2"]
    _, summary, table = app.run_detection(np.full((480, 640, 3), 128, np.uint8), "YOLOv8n", 0.25, 0.7, ["Car"])
    assert "No vehicles detected" in summary and table.empty
    for bad in (None, np.zeros((10, 10, 3), np.uint8)):
        with pytest.raises(gr.Error):
            app.run_detection(bad, "YOLOv8n", 0.25, 0.7, ["Car"])
