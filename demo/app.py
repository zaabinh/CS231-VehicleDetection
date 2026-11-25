"""Gradio UI for the vehicle detection demo. Inference lives in inference.py."""

import logging
from pathlib import Path

import gradio as gr
import pandas as pd

from inference import COLORS, DEFAULT_MODEL, DEVICE, MODEL_SPECS, count_by_class, detect, draw, get_model, text_color, validate_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vehicle-demo")

REPO_URL = "https://github.com/zaabinh/CS231-VehicleCounting"
EXAMPLES = sorted(str(p) for p in (Path(__file__).parent / "examples").glob("*.jpg"))
DISPLAY_ORDER = ("Car", "Motorbike", "Truck", "Bus")

# Reported in the project README (test split, 100 images). Not re-measured by this demo.
BENCHMARK = pd.DataFrame(
    [
        ("HOG + SVM", 4.30, 0.62, 0.12, 0.03, 5.00),
        ("SSD300", 38.70, 38.60, 79.20, 54.60, 26.00),
        ("Faster R-CNN", 39.62, 39.81, 91.79, 67.43, 12.21),
        ("YOLOv8n", 94.56, 90.72, 95.38, 76.19, 142.68),
    ],
    columns=["Model", "Precision (%)", "Recall (%)", "mAP@50 (%)", "mAP@50:95 (%)", "FPS (reported)"],
)

HEADER = f"""
# Vehicle Detection Benchmark
### Computer Vision for Traffic Surveillance

Detects **cars, motorbikes, trucks and buses** in traffic-camera images with models trained for the
CS231 (Introduction to Computer Vision) course project at UIT – VNU-HCM. Pick an example or upload a photo,
then press **Run detection**. Every result below is real model inference running on this server ({DEVICE.upper()}).

[GitHub repository]({REPO_URL}) · Default model: **YOLOv8n fine-tuned on the project dataset** (`models/yolov8.pt`)
"""

MODEL_INFO = f"""
| | |
|---|---|
| **Model variant** | YOLOv8n (~3.0 M parameters), Ultralytics |
| **Checkpoint** | `models/yolov8.pt` – fine-tuned from COCO-pretrained `yolov8n.pt` (checkpoint `train_args`: 100 epochs, imgsz 832, SGD, run `yolov8_main_online_aug`) |
| **Classes** | 0 bus · 1 car · 2 motorbike · 3 truck (read from the checkpoint and `data/main/data.yaml`) |
| **Dataset** | [Roboflow Vehicle Detection v1](https://universe.roboflow.com/skripsi-1qzlz/vehicle-detection-bvxpr) (CC BY 4.0) – 1,000 CCTV images from Yogyakarta, Indonesia; 800 train / 100 valid / 100 test; images pre-resized to 640x640 |
| **Input** | RGB image, letterboxed to 832x832 (aspect ratio preserved) by Ultralytics |
| **Defaults** | confidence 0.25 (used for the project's test predictions), NMS IoU 0.7 (Ultralytics default, stored in the checkpoint) |
| **Inference device** | {DEVICE.upper()} |
| **Method** | Single-stage, anchor-free detector: CSP/C2f backbone + SPPF, multi-scale head predicting boxes and class scores, followed by NMS (built into Ultralytics; not re-applied here) |

**Also selectable:** Faster R-CNN (ResNet-50 FPN, `best_FasterRCNN.pth`) and SSD300 (VGG16, `best_SSD.pth`),
fine-tuned by the team on the same data. They use the preprocessing from their training notebooks
(stretched resize to 640 / 300 px + ImageNet normalization) and torchvision's built-in NMS, so the IoU slider
applies to YOLOv8 only. Faster R-CNN takes several seconds per image on CPU.
HOG + SVM is shown in the benchmark only: its live pipeline is slow and its detections are near zero (mAP@50 0.12%).

Example images are from the dataset's **test split**, which was not used for training.
"""

BENCHMARK_NOTES = """
**These are previously reported results from the project's notebooks, not measurements from this demo.**
The live latency shown in the detection summary is measured on this server and is unrelated to these FPS values.

**Cross-check against notebook outputs**
- YOLOv8 Precision / Recall / mAP match the test-split evaluation of `models/yolov8.pt` in
  `notebooks/YOLOv8/yolov8_aug_main.ipynb`. That notebook prints **131.89 FPS**, not 142.68; the 142.68 value
  could not be traced to a saved output.
- Faster R-CNN notebook output: P 39.37%, R 39.48%, mAP@50 91.79%, mAP@50:95 67.26%, 12.21 FPS (small P/R/mAP@50:95 differences from the table).
- SSD notebook output: P 38.07%, R 38.06% (table: 38.70 / 38.60, likely a transcription error); mAP and FPS match.
- HOG + SVM notebook output: P 3.85%, R 0.97%, mAP@50 0.14%, mAP@50:95 0.04%, 5.00 FPS.

**Comparability caveats**
- All four were evaluated on the same 100-image test split of `data/main`.
- *Precision/Recall are not computed the same way.* YOLOv8 (Ultralytics) and HOG+SVM report detection P/R at IoU 0.5.
  SSD and Faster R-CNN report a macro-averaged *classification* score over predicted vs. ground-truth labels, which
  explains why their P/R look low next to high mAP@50. Compare these models on **mAP** rather than P/R.
- mAP: Ultralytics `val()` for YOLOv8; `torchmetrics` MeanAveragePrecision for SSD / Faster R-CNN; a custom VOC-style AP for HOG+SVM.
- Input size differs: YOLOv8 832 px letterbox; Faster R-CNN 640 px (then torchvision's internal resize); SSD 300 px; HOG 48x48 sliding windows.
- FPS: YOLOv8 = 1000 / Ultralytics forward-pass time (batch 16, Colab Tesla T4, excludes pre/post-processing);
  SSD / Faster R-CNN = full `model()` call incl. internal preprocessing and NMS (batch 4, Colab GPU);
  HOG+SVM = full sliding-window pipeline per image. The FPS values are therefore indicative only.
"""

METHODOLOGY = """
**Traffic image → Preprocessing (resize / letterbox, normalize) → YOLOv8 object detection → Confidence filtering + NMS → Bounding boxes and class labels → Per-class detection counts**

- **Classification vs. detection.** Image classification assigns one label to a whole image. Object detection finds
  *every* object, telling you *what* it is and *where* it is.
- **Bounding box.** The rectangle (x1, y1, x2, y2) in pixels that localizes one detected object.
- **Confidence threshold.** Each box has a score in [0, 1]. Boxes below the threshold are discarded:
  raising it gives fewer, more certain detections; lowering it finds more objects but adds false positives.
- **Intersection over Union (IoU).** Overlap area ÷ union area of two boxes. Non-maximum suppression (NMS) drops a box
  whose IoU with a higher-scoring box of the same class exceeds the IoU threshold, so each object is counted once.
- **Precision and recall.** Precision = share of predicted boxes that are correct; recall = share of real objects that were found.
  A prediction counts as correct when its IoU with a ground-truth box of the same class is at least 0.5.
- **mAP@50 and mAP@50:95.** Average precision summarizes the precision–recall curve for one class; mAP averages it over
  classes. mAP@50 uses IoU ≥ 0.5; mAP@50:95 averages over IoU thresholds 0.50–0.95 and therefore also rewards tight boxes.

Counts are **objects detected in the current image**, not unique vehicles tracked over time.
"""

CSS = ".legend span{display:inline-block;padding:2px 8px;margin-right:6px;border-radius:4px;font-weight:600}"
LEGEND = "<div class='legend'>" + "".join(
    f"<span style='background:rgb{COLORS[c]};color:rgb{text_color(COLORS[c])}'>{c}</span>" for c in DISPLAY_ORDER
) + "</div>"


def summary_markdown(result, conf):
    counts = count_by_class(result.detections)
    head = "| Total | " + " | ".join(f"{c}s" if c != "Bus" else "Buses" for c in DISPLAY_ORDER) + " | Model inference | End-to-end |"
    row = f"| **{len(result.detections)}** | " + " | ".join(str(counts[c]) for c in DISPLAY_ORDER)
    row += f" | {result.inference_ms:.0f} ms | {result.total_ms:.0f} ms |"
    note = f"\n\nModel: **{result.model}** on **{result.device.upper()}** · confidence ≥ {conf:.2f}. "
    note += "*Model inference* = forward pass only; *end-to-end* adds preprocessing and postprocessing."
    if not result.detections:
        note += "\n\n**No vehicles detected** at this threshold. Try lowering the confidence threshold."
    return f"{head}\n|{'---|' * 7}\n{row}{note}"


def run_detection(image, model_name, conf, iou, classes):
    if image is None:
        raise gr.Error("Please upload an image or choose an example first.")
    if not classes:
        raise gr.Error("Select at least one vehicle class.")
    try:
        result = detect(image, model_name, conf=conf, iou=iou, classes=classes)
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception:
        log.exception("Inference failed")
        raise gr.Error("Inference failed on the server. Please try another image.")
    rows = [
        [i, d.class_name, round(d.confidence, 3), *(round(v) for v in d.box)] for i, d in enumerate(result.detections, 1)
    ]
    table = pd.DataFrame(rows, columns=["ID", "Class", "Confidence", "x1", "y1", "x2", "y2"])
    return draw(validate_image(image), result.detections), summary_markdown(result, conf), table


def on_model_change(model_name):
    spec = MODEL_SPECS[model_name]
    return gr.update(value=spec.default_conf), gr.update(interactive=spec.supports_iou)


def build():
    with gr.Blocks(title="Vehicle Detection Benchmark", analytics_enabled=False) as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            original = gr.Image(label="Original image", type="numpy", image_mode="RGB", sources=["upload", "clipboard"], height=420)
            annotated = gr.Image(label="Detections", type="numpy", interactive=False, height=420)
        gr.HTML(LEGEND)
        summary = gr.Markdown("Run detection to see per-class counts and latency.")
        with gr.Row():
            with gr.Column(scale=2):
                model = gr.Dropdown(list(MODEL_SPECS), value=DEFAULT_MODEL, label="Model")
                conf = gr.Slider(0.05, 0.95, value=MODEL_SPECS[DEFAULT_MODEL].default_conf, step=0.05, label="Confidence threshold")
                iou = gr.Slider(0.1, 0.95, value=0.7, step=0.05, label="IoU (NMS) threshold – YOLOv8 only")
                classes = gr.CheckboxGroup(list(DISPLAY_ORDER), value=list(DISPLAY_ORDER), label="Vehicle classes to show")
                with gr.Row():
                    run_btn = gr.Button("Run detection", variant="primary")
                    clear_btn = gr.ClearButton(value="Clear")
            with gr.Column(scale=3):
                gr.Examples(EXAMPLES, inputs=original, label="Example images (dataset test split)")
        details = gr.Dataframe(label="Detection details (pixel coordinates in the original image)", interactive=False)
        clear_btn.add([original, annotated, details])
        clear_btn.click(lambda: "Run detection to see per-class counts and latency.", outputs=summary)

        run_btn.click(run_detection, [original, model, conf, iou, classes], [annotated, summary, details])
        model.change(on_model_change, model, [conf, iou])

        with gr.Tab("Model information"):
            gr.Markdown(MODEL_INFO)
        with gr.Tab("Experimental benchmark"):
            gr.Dataframe(BENCHMARK, interactive=False, label="Reported test-split results (not re-measured)")
            with gr.Row():
                for metric in ("mAP@50 (%)", "mAP@50:95 (%)", "FPS (reported)"):
                    gr.BarPlot(BENCHMARK, x="Model", y=metric, title=metric, sort=list(BENCHMARK["Model"]),
                                x_label_angle=-30, y_lim=None if "FPS" in metric else [0, 100], height=300)
            gr.Markdown(
                "**Trade-off.** In these runs YOLOv8n was both the most accurate (mAP@50:95 76.2%) and the fastest. "
                "Faster R-CNN came second on accuracy but was the slowest deep model; SSD300 was faster than Faster R-CNN "
                "but less accurate; hand-crafted HOG + SVM features did not transfer to dense CCTV scenes. "
                "This is one dataset, one training run per model and non-identical FPS measurements, so treat the ranking as indicative."
            )
            gr.Markdown(BENCHMARK_NOTES)
        with gr.Tab("Methodology"):
            gr.Markdown(METHODOLOGY)
    return demo


if __name__ == "__main__":
    get_model(DEFAULT_MODEL)  # load once at startup so the first request is not slow
    build().queue(max_size=16, default_concurrency_limit=1).launch(
        max_file_size="10mb", show_error=False, theme=gr.themes.Soft(), css=CSS
    )
