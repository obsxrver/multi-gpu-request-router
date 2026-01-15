import asyncio
import json
import os
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

COMFYUI_PATH = Path(os.getenv("COMFYUI_PATH", "/opt/ComfyUI"))
COMFYUI_MAIN = COMFYUI_PATH / "main.py"
COMFYUI_INPUT_DIR = Path(os.getenv("COMFYUI_INPUT_DIR", str(COMFYUI_PATH / "input")))

GPU_PORT_BASE = int(os.getenv("COMFYUI_PORT_BASE", "8188"))
COMFYUI_ARGS = os.getenv("COMFYUI_ARGS", "").split()

POLL_SECONDS = float(os.getenv("PROMPT_POLL_SECONDS", "2"))
POLL_TIMEOUT = float(os.getenv("PROMPT_POLL_TIMEOUT", "1200"))

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@dataclass
class Job:
    job_id: str
    workflow: Dict[str, Any]
    image_filename: str
    created_at: datetime
    status: str = "queued"
    gpu_id: Optional[int] = None
    prompt_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GPUWorker:
    gpu_id: int
    port: int
    process: Optional[subprocess.Popen] = None
    busy: bool = False
    last_error: Optional[str] = None

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not COMFYUI_MAIN.exists():
            raise RuntimeError(f"ComfyUI not found at {COMFYUI_MAIN}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        cmd = [
            shutil.which("python") or "python",
            str(COMFYUI_MAIN),
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        cmd.extend(COMFYUI_ARGS)
        self.process = subprocess.Popen(
            cmd,
            cwd=str(COMFYUI_PATH),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)


class GPUManager:
    def __init__(self, gpu_ids: list[int]):
        self.gpus = [GPUWorker(gpu_id=gpu_id, port=GPU_PORT_BASE + idx) for idx, gpu_id in enumerate(gpu_ids)]
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.jobs: Dict[str, Job] = {}
        self.dispatch_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if not self.dispatch_task:
            self.dispatch_task = asyncio.create_task(self.dispatch_loop())

    async def dispatch_loop(self) -> None:
        while True:
            job = await self.queue.get()
            gpu = await self.wait_for_free_gpu()
            await self.run_job(gpu, job)
            self.queue.task_done()

    async def wait_for_free_gpu(self) -> GPUWorker:
        while True:
            for gpu in self.gpus:
                if not gpu.busy:
                    return gpu
            await asyncio.sleep(0.5)

    async def run_job(self, gpu: GPUWorker, job: Job) -> None:
        gpu.busy = True
        job.status = "running"
        job.gpu_id = gpu.gpu_id
        try:
            gpu.start()
            prompt_id = await submit_prompt(gpu, job.workflow)
            job.prompt_id = prompt_id
            await wait_for_prompt_completion(gpu, prompt_id)
            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            gpu.last_error = str(exc)
        finally:
            gpu.busy = False

    def enqueue(self, job: Job) -> None:
        self.jobs[job.job_id] = job
        self.queue.put_nowait(job)


async def submit_prompt(gpu: GPUWorker, workflow: Dict[str, Any]) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{gpu.base_url()}/prompt", json={"prompt": workflow})
        response.raise_for_status()
        data = response.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"Unexpected response: {data}")
        return data["prompt_id"]


async def wait_for_prompt_completion(gpu: GPUWorker, prompt_id: str) -> None:
    deadline = datetime.now(tz=timezone.utc).timestamp() + POLL_TIMEOUT
    async with httpx.AsyncClient(timeout=30) as client:
        while datetime.now(tz=timezone.utc).timestamp() < deadline:
            response = await client.get(f"{gpu.base_url()}/history/{prompt_id}")
            if response.status_code == 404:
                await asyncio.sleep(POLL_SECONDS)
                continue
            response.raise_for_status()
            data = response.json()
            if data:
                return
            await asyncio.sleep(POLL_SECONDS)
    raise TimeoutError("Prompt did not complete before timeout.")


def parse_gpu_ids() -> list[int]:
    raw = os.getenv("CUDA_VISIBLE_DEVICES")
    if raw:
        ids = [int(value) for value in raw.split(",") if value.strip().isdigit()]
        if ids:
            return ids
    count = int(os.getenv("GPU_COUNT", "1"))
    return list(range(count))


def update_workflow_image(workflow: Dict[str, Any], image_filename: str) -> Dict[str, Any]:
    updated = json.loads(json.dumps(workflow))
    for node in updated.values():
        if node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})
            node["inputs"]["image"] = image_filename
            return updated
    raise ValueError("Workflow must include a LoadImage node.")


@app.on_event("startup")
async def startup_event() -> None:
    app.state.gpu_manager = GPUManager(parse_gpu_ids())
    app.state.gpu_manager.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    manager: GPUManager = app.state.gpu_manager
    for gpu in manager.gpus:
        gpu.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    manager: GPUManager = app.state.gpu_manager
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "gpus": manager.gpus,
            "jobs": list(manager.jobs.values())[-20:],
        },
    )


@app.post("/submit")
async def submit_job(
    workflow_json: str = Form(...),
    image: UploadFile = File(...),
) -> RedirectResponse:
    manager: GPUManager = app.state.gpu_manager
    try:
        workflow = json.loads(workflow_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
    image_filename = f"{uuid.uuid4().hex}_{image.filename}"
    COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = COMFYUI_INPUT_DIR / image_filename
    with destination.open("wb") as buffer:
        buffer.write(await image.read())
    try:
        updated_workflow = update_workflow_image(workflow, image_filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    job = Job(
        job_id=str(uuid.uuid4()),
        workflow=updated_workflow,
        image_filename=image_filename,
        created_at=datetime.now(tz=timezone.utc),
    )
    manager.enqueue(job)
    return RedirectResponse(url=f"/jobs/{job.job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_status(job_id: str, request: Request) -> HTMLResponse:
    manager: GPUManager = app.state.gpu_manager
    job = manager.jobs.get(job_id)
    if not job:
        return HTMLResponse("Job not found", status_code=404)
    return TEMPLATES.TemplateResponse(
        "job.html",
        {
            "request": request,
            "job": job,
        },
    )
