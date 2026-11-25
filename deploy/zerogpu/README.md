---
title: Vehicle Detection Benchmark
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: demo/space.py
pinned: false
short_description: Compare trained YOLOv8, Faster R-CNN and SSD vehicle detectors
---

# Vehicle Detection Benchmark

Detect and count cars, motorbikes, trucks and buses in traffic images using the project's
trained YOLOv8n, Faster R-CNN and SSD300 checkpoints. Select a model, upload an image or
choose a test-split example, and compare annotated detections, counts and measured latency.

[Source and project README](https://github.com/zaabinh/CS231-VehicleCounting).

Runs on free ZeroGPU: requests may queue and GPU access has daily quotas.
Reported benchmark values come from the team's experiments, not this deployment.
Counts refer to a single image, not tracked unique vehicles across video frames.

Example images: [Roboflow Vehicle Detection v1](https://universe.roboflow.com/skripsi-1qzlz/vehicle-detection-bvxpr),
CC BY 4.0, traffic camera images from Yogyakarta, Indonesia.
