import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nst.image_validation import ImageValidationError, validate_image_bytes
from nst.settings import DEFAULT_STEPS

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per image
MAX_IMAGE_PIXELS = 25_000_000
UPLOAD_CHUNK_BYTES = 1024 * 1024
RESULT_RETENTION_DAYS = 30
UPLOAD_RETENTION_HOURS = 24

logger = logging.getLogger(__name__)
progress_lock = Lock()
render_lock = asyncio.Lock()
progress_state = {
    "status": "idle",
    "progress": 0,
    "step": 0,
    "steps": DEFAULT_STEPS,
    "loss": None,
    "message": "Choose two images to begin.",
}


def cleanup_expired_results():
    """Remove only this app's PNG outputs older than the retention period."""
    cutoff = datetime.now(timezone.utc).timestamp() - (
        RESULT_RETENTION_DAYS * 24 * 60 * 60
    )
    for result_path in RESULTS_DIR.glob("*.png"):
        try:
            if result_path.stat().st_mtime < cutoff:
                result_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove expired result %s", result_path)


def cleanup_stale_uploads():
    """Remove temporary upload files left behind by an interrupted process."""
    cutoff = datetime.now(timezone.utc).timestamp() - (
        UPLOAD_RETENTION_HOURS * 60 * 60
    )
    for upload_path in UPLOADS_DIR.iterdir():
        try:
            if upload_path.is_file() and upload_path.stat().st_mtime < cutoff:
                upload_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove stale upload %s", upload_path)


@asynccontextmanager
async def lifespan(_app):
    cleanup_expired_results()
    cleanup_stale_uploads()
    yield


app = FastAPI(title="StyleLab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


async def read_valid_image(upload: UploadFile, label: str) -> bytes:
    """Read an upload in bounded chunks and validate its actual image data."""
    image_bytes = bytearray()
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        if len(image_bytes) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{label} image must be 10 MB or smaller.",
            )
        image_bytes.extend(chunk)

    try:
        validate_image_bytes(
            bytes(image_bytes),
            label,
            max_bytes=MAX_UPLOAD_BYTES,
            max_pixels=MAX_IMAGE_PIXELS,
        )
    except ImageValidationError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error),
        ) from error

    return bytes(image_bytes)


def update_progress(job_id, *, status=None, step=None, steps=None, loss=None,
                    message=None):
    """Update progress safely from either the async route or worker thread."""
    with progress_lock:
        if progress_state.get("job_id") != job_id:
            return

        if status is not None:
            progress_state["status"] = status
        if step is not None:
            progress_state["step"] = step
        if steps is not None:
            progress_state["steps"] = steps
        if loss is not None:
            progress_state["loss"] = round(loss, 4)
        if step is not None and steps:
            progress_state["progress"] = round(100 * step / steps)
        if message is not None:
            progress_state["message"] = message


def progress_snapshot():
    with progress_lock:
        return dict(progress_state)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/progress")
def get_progress():
    return progress_snapshot()


@app.post("/api/stylize")
async def stylize(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    strength: float = Form(1.0, ge=0.1, le=1.0),
):
    # Style transfer is CPU-heavy. Refuse overlapping requests so concurrent
    # browser tabs cannot make the laptop run several renders at once.
    if render_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A render is already running. Please wait for it to finish.",
        )

    async with render_lock:
        job_id = uuid4().hex
        content_path = UPLOADS_DIR / f"{job_id}-content"
        style_path = UPLOADS_DIR / f"{job_id}-style"
        result_path = RESULTS_DIR / f"{job_id}.png"

        with progress_lock:
            progress_state.clear()
            progress_state.update({
                "job_id": job_id,
                "status": "validating",
                "progress": 0,
                "step": 0,
                "steps": DEFAULT_STEPS,
                "loss": None,
                "message": "Checking your images…",
            })

        try:
            cleanup_expired_results()
            cleanup_stale_uploads()
            content_bytes = await read_valid_image(content, "Content")
            style_bytes = await read_valid_image(style, "Style")
            content_path.write_bytes(content_bytes)
            style_path.write_bytes(style_bytes)

            with progress_lock:
                progress_state.update({
                    "status": "running",
                    "message": "Preparing the neural style-transfer model…",
                })

            def report_step(step, total_steps, loss):
                update_progress(
                    job_id,
                    status="running",
                    step=step,
                    steps=total_steps,
                    loss=loss,
                    message=f"Creating your artwork… step {step} of {total_steps}",
                )

            # Import lazily so server routes and upload-validation tests do not
            # need to load the large VGG19 model just to start the web app.
            from nst.style_transfer import stylize_images

            await run_in_threadpool(
                stylize_images,
                content_path,
                style_path,
                result_path,
                strength,
                DEFAULT_STEPS,
                progress_callback=report_step,
            )

            update_progress(
                job_id,
                status="completed",
                step=DEFAULT_STEPS,
                steps=DEFAULT_STEPS,
                message="Your artwork is ready.",
            )
            return {"image_url": f"/results/{result_path.name}"}

        except HTTPException as error:
            update_progress(
                job_id,
                status="failed",
                message=str(error.detail),
            )
            raise
        except Exception as error:
            logger.exception("Style transfer failed")
            update_progress(
                job_id,
                status="failed",
                message="Style transfer failed. Check the server terminal for details.",
            )
            result_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="Style transfer failed. Check the server terminal for details.",
            ) from error
        finally:
            content_path.unlink(missing_ok=True)
            style_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
