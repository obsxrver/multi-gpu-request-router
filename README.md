# Multi-GPU ComfyUI Request Router

This project provides a simple web interface that manages one ComfyUI process per CUDA device and routes incoming workflow requests to the first available GPU.

## Features

- Starts a dedicated ComfyUI server per GPU (ports start at `8188`).
- Upload a workflow JSON and an input image.
- Automatically rewrites the `LoadImage` node to use the uploaded image.
- Queues jobs and dispatches them to the first free GPU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure ComfyUI is installed and `COMFYUI_PATH` points to its directory.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to submit workflows.

## Environment variables

- `COMFYUI_PATH`: Path to the ComfyUI checkout (default `/opt/ComfyUI`).
- `COMFYUI_INPUT_DIR`: Override the ComfyUI input directory.
- `COMFYUI_PORT_BASE`: Starting port for GPU servers (default `8188`).
- `COMFYUI_ARGS`: Extra arguments passed to ComfyUI on launch.
- `GPU_COUNT`: Number of GPUs to assume if `CUDA_VISIBLE_DEVICES` is not set.
- `PROMPT_POLL_SECONDS`: Seconds between prompt status checks.
- `PROMPT_POLL_TIMEOUT`: Timeout in seconds before marking a prompt failed.
