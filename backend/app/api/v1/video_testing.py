"""
NETRA — Live Video Testing API Router
Module: backend/app/api/v1/video_testing.py
=============================================================================
Manages test video uploads, AI Engine processing jobs, and real-time telemetry:
1. POST /jobs         -> Upload video, validate camera, spawn AI worker
2. GET  /jobs/{id}    -> Poll real-time processing status & AI metrics
3. GET  /jobs/{id}/video -> Stream transcoded H.264 annotated video
=============================================================================
"""

import os
import sys
import json
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.session import get_db, SessionLocal
from app.models.entities import Camera
from app.schemas.ai_contracts import VehicleTrackInput
from app.services.ingest_service import ingest_tracks

logger = logging.getLogger("NETRA.VideoTestingAPI")

router = APIRouter(prefix="/video-testing", tags=["Live Video Testing"])

# Resolve Project Root and Output Directories
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
UPLOADS_DIR = PROJECT_ROOT / "data" / "test_uploads"
RUNS_DIR = PROJECT_ROOT / "runs" / "video_testing"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_BYTES = 150 * 1024 * 1024  # 150 MB


def _resolve_python_executable() -> Path:
    """Resolve the AI Engine Python executable (preferring root venv with PyTorch & CUDA)."""
    root_venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if root_venv_python.exists():
        return root_venv_python
    backend_venv_python = BACKEND_DIR / ".venv" / "Scripts" / "python.exe"
    if backend_venv_python.exists():
        return backend_venv_python
    return Path(sys.executable)


def _process_video_background_task(job_id: str, camera_id: str, input_video_path: Path, job_dir: Path):
    """Background execution worker: invokes AI pipeline script and performs DB ingestion."""
    logger.info(f"Starting background AI execution for job {job_id} on camera {camera_id}...")
    state_file = job_dir / "job_state.json"
    python_exe = _resolve_python_executable()
    runner_script = PROJECT_ROOT / "scripts" / "run_video_testing_pipeline.py"

    cmd = [
        str(python_exe),
        str(runner_script),
        "--job-id", str(job_id),
        "--camera-id", str(camera_id),
        "--video-path", str(input_video_path),
        "--output-dir", str(job_dir),
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if proc.returncode != 0:
            logger.error(f"AI runner failed with exit code {proc.returncode}:\n{proc.stderr}\n{proc.stdout}")
            _write_state(
                state_file,
                status="FAILED",
                stage="FAILED",
                stage_label="AI Engine execution failed. Check backend logs.",
                progress=100,
                error=proc.stderr or proc.stdout or "Subprocess returned non-zero code.",
            )
            return

        # Perform Backend-Side Database Ingestion (using native backend SessionLocal)
        telemetry_file = job_dir / "unified_vehicle_tracks.json"
        if telemetry_file.exists():
            with open(telemetry_file, "r", encoding="utf-8") as tf:
                telemetry = json.load(tf)

            raw_tracks = telemetry.get("vehicle_tracks", [])
            validated_tracks = []
            for t in raw_tracks:
                try:
                    validated_tracks.append(VehicleTrackInput.model_validate(t))
                except Exception as ve:
                    logger.debug(f"Skipping track validation: {ve}")

            if validated_tracks:
                with SessionLocal() as db:
                    count = ingest_tracks(db, camera_id=camera_id, tracks=validated_tracks, commit=True)
                    logger.info(f"Job {job_id}: Successfully ingested {count} observations into PostgreSQL for {camera_id}.")

                    # Update db_ingested_observations in job_state.json
                    if state_file.exists():
                        try:
                            with open(state_file, "r", encoding="utf-8") as sf:
                                state_data = json.load(sf)
                            if "metrics" in state_data:
                                state_data["metrics"]["db_ingested_observations"] = count
                            with open(state_file, "w", encoding="utf-8") as sf:
                                json.dump(state_data, sf, indent=2)
                        except Exception as update_err:
                            logger.warning(f"Could not update state file ingest count: {update_err}")

    except subprocess.TimeoutExpired:
        logger.error(f"Job {job_id} timed out after 300 seconds.")
        _write_state(
            state_file,
            status="FAILED",
            stage="FAILED",
            stage_label="AI processing timed out after 5 minutes.",
            progress=100,
            error="Processing timeout exceeded.",
        )
    except Exception as e:
        logger.error(f"Unexpected error in background worker for {job_id}: {e}", exc_info=True)
        _write_state(
            state_file,
            status="FAILED",
            stage="FAILED",
            stage_label="Unexpected execution failure.",
            progress=100,
            error=str(e),
        )


def _write_state(state_file: Path, status: str, stage: str, stage_label: str, progress: int, error: Optional[str] = None):
    """Helper to update job state file, preserving existing job metadata."""
    existing = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing.update({
        "status": status,
        "stage": stage,
        "stage_label": stage_label,
        "progress_percent": progress,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    })
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write state file {state_file}: {e}")


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_video_testing_job(
    background_tasks: BackgroundTasks,
    camera_id: str = Form(..., description="Camera ID e.g. CAM_01"),
    file: UploadFile = File(..., description="Test traffic video (MP4, AVI, MOV)"),
    db: Session = Depends(get_db),
):
    """
    Upload a traffic video file, validate camera association, and trigger the real NETRA AI Engine pipeline.
    """
    # 1. Validate Camera Existence in DB
    camera = db.get(Camera, camera_id)
    if not camera:
        # Check case-insensitive match
        stmt = select(Camera).where(Camera.id.ilike(camera_id))
        camera = db.execute(stmt).scalars().first()

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found in database. Please select a registered camera node.",
        )

    # 2. Validate File Extension
    raw_filename = file.filename or "uploaded_video.mp4"
    file_ext = Path(raw_filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{file_ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 3. Create Unique Job Directories
    job_id = f"VT-{uuid.uuid4().hex[:10].upper()}"
    job_upload_dir = UPLOADS_DIR / job_id
    job_run_dir = RUNS_DIR / job_id

    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_run_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_upload_dir / f"input{file_ext}"

    # 4. Stream Upload to Disk and Check Size
    file_size = 0
    try:
        with open(input_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    shutil.rmtree(job_upload_dir, ignore_errors=True)
                    shutil.rmtree(job_run_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to receive video stream.",
        )

    # 5. Initialize Job State
    state_file = job_run_dir / "job_state.json"
    initial_state = {
        "job_id": job_id,
        "camera_id": camera.id,
        "camera_name": camera.name,
        "road_name": camera.road_name,
        "location_name": camera.location_name,
        "filename": raw_filename,
        "file_size_bytes": file_size,
        "status": "QUEUED",
        "stage": "QUEUED",
        "stage_label": "Video received. Allocating AI execution worker...",
        "progress_percent": 5,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
        "metrics": {},
        "video_relative_url": f"/api/v1/video-testing/jobs/{job_id}/video",
    }
    with open(state_file, "w", encoding="utf-8") as sf:
        json.dump(initial_state, sf, indent=2)

    # 6. Dispatch Background AI Worker
    background_tasks.add_task(
        _process_video_background_task,
        job_id=job_id,
        camera_id=camera.id,
        input_video_path=input_path,
        job_dir=job_run_dir,
    )

    logger.info(f"Created video testing job {job_id} for camera {camera.id} ({raw_filename}, {file_size / 1024 / 1024:.2f} MB)")
    return {
        "job_id": job_id,
        "camera_id": camera.id,
        "camera_name": camera.name,
        "status": "QUEUED",
        "filename": raw_filename,
        "message": "AI processing job created and queued successfully.",
    }


@router.get("/jobs/{job_id}")
def get_video_testing_job_status(job_id: str):
    """
    Poll the current processing state, stage, progress, and AI metrics for a given job.
    """
    job_run_dir = RUNS_DIR / job_id
    state_file = job_run_dir / "job_state.json"

    if not state_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading job state for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading job state.",
        )


@router.get("/jobs/{job_id}/video")
def get_processed_video(job_id: str):
    """
    Stream the annotated AI output video with detection and tracking overlays.
    """
    job_run_dir = RUNS_DIR / job_id
    web_video = job_run_dir / "annotated_video.mp4"
    raw_video = job_run_dir / "raw_annotated.mp4"

    target_video = web_video if web_video.exists() else raw_video
    if not target_video.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed video is not ready yet or was not generated.",
        )

    return FileResponse(
        path=str(target_video),
        media_type="video/mp4",
        filename=f"{job_id}_annotated.mp4",
    )
