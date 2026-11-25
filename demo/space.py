"""ZeroGPU entry point; keep the local Gradio application reusable."""
import spaces  # Initialize CUDA emulation before importing torch through inference.
import logging

# ZeroGPU's internal HTTP URLs can include temporary allocation credentials.
logging.getLogger("httpx").setLevel(logging.WARNING)

from inference import MODEL_SPECS, detect, get_model

for name in MODEL_SPECS:
    get_model(name, warmup=False)

import app

app.detect = spaces.GPU(duration=20)(detect)

if __name__ == "__main__":
    app.build().queue(max_size=16, default_concurrency_limit=1).launch(
        max_file_size="10mb", show_error=False, theme=app.gr.themes.Soft(), css=app.CSS,
    )
