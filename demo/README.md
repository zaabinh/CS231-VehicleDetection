# Vehicle detection demo

The Gradio UI serves the project's YOLOv8n, Faster R-CNN and SSD300 checkpoints.
HOG + SVM appears only in the reported benchmark. Example images are from the test split.

Local run: `pip install -r demo/requirements.txt` then `python demo/app.py`.
Tests: `python -m pytest demo/tests` (requires pytest and real model artifacts).
Run `git lfs pull` first if model files are pointers.

Publish on free Hugging Face ZeroGPU:

```powershell
python deploy/zerogpu/publish.py --token-name deploying
```

The publisher uploads only app code, example images and the three required checkpoints.
The Space uses its own README and requirements from `deploy/zerogpu/`; the project root
README is preserved exactly as fetched from origin. All model weights are loaded at startup,
without GPU warmup; detection runs inside a ZeroGPU-decorated function. Requests are serialized.
ZeroGPU access has queues and daily quotas. Uploads are limited to 10 MB and 32–4096 pixels per side.
