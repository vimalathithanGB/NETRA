"""
NETRA — Live Video Testing AI Pipeline Runner
Module: scripts/run_video_testing_pipeline.py
=============================================================================
Executes the real NETRA AI Engine pipeline on an uploaded test video:
1. Video validation & extraction
2. Vehicle Detection (YOLOv8n UVH-26)
3. ByteTrack Multi-Object Tracking
4. License Plate Detection (YOLOv8n Plate Detector)
5. PaddleOCR with Confidence Voting & Cleaning
6. OSNet-AIN Vehicle Re-ID Feature Extraction
7. PostgreSQL / PostGIS Database Ingestion (via existing ingest_service)
8. Web-compatible H.264 video transcoding (via FFmpeg)
9. Real telemetry state reporting (job_state.json)
=============================================================================
"""

import os
import sys
import json
import time
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Ensure project root and backend are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [NETRA-VideoRunner] %(message)s"
)
logger = logging.getLogger("NETRA-VideoRunner")


def update_job_state(
    state_file: Path,
    status: str,
    stage: str,
    stage_label: str,
    progress_percent: int,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    video_relative_url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """Write current execution state to job_state.json safely, preserving existing metadata."""
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
        "progress_percent": progress_percent,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    })
    if metrics is not None:
        existing["metrics"] = metrics
    if video_relative_url is not None:
        existing["video_relative_url"] = video_relative_url
    if extra:
        existing.update(extra)

    temp_file = state_file.with_suffix(".tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        shutil.move(str(temp_file), str(state_file))
    except Exception as e:
        logger.warning(f"Failed to write job state to {state_file}: {e}")


def transcode_to_web_mp4(input_path: Path, output_path: Path) -> bool:
    """Transcode raw OpenCV mp4v video to H.264 (avc1 / yuv420p) for HTML5 browser playback."""
    logger.info(f"Transcoding {input_path} to browser-compatible H.264 at {output_path}...")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            logger.info("FFmpeg H.264 transcoding succeeded.")
            return True
        else:
            logger.warning(f"FFmpeg failed with code {res.returncode}: {res.stderr}")
    except Exception as e:
        logger.warning(f"FFmpeg invocation failed: {e}")

    # Fallback: copy original if transcoding fails
    try:
        if not output_path.exists() or output_path.stat().st_size == 0:
            shutil.copy2(str(input_path), str(output_path))
    except Exception:
        pass
    return False


def run_pipeline(
    job_id: str,
    camera_id: str,
    video_path: Path,
    output_dir: Path,
    max_frames: Optional[int] = None,
):
    """Executes the full pipeline and handles state transitions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "job_state.json"

    # Initial state: QUEUED / INITIALIZING
    update_job_state(
        state_file=state_file,
        status="PROCESSING",
        stage="VIDEO_LOADING",
        stage_label="Validating video file and allocating GPU/CPU workers...",
        progress_percent=10,
        extra={"job_id": job_id, "camera_id": camera_id, "video_name": video_path.name},
    )

    raw_annotated_video = output_dir / "raw_annotated.mp4"
    web_annotated_video = output_dir / "annotated_video.mp4"
    telemetry_json = output_dir / "unified_vehicle_tracks.json"

    try:
        # Step 1: Import UnifiedVehiclePipeline
        logger.info("Initializing UnifiedVehiclePipeline...")
        from tracking.unified_vehicle_pipeline import UnifiedVehiclePipeline

        update_job_state(
            state_file=state_file,
            status="PROCESSING",
            stage="MODEL_INITIALIZATION",
            stage_label="Loading YOLOv8n UVH-26, Plate Detector, PaddleOCR & OSNet-AIN...",
            progress_percent=20,
        )

        pipeline = UnifiedVehiclePipeline(
            output_dir=str(output_dir),
            vehicle_conf=0.40,
            plate_conf=0.35,
            ocr_conf=0.50,
            ocr_interval=5,
            reid_interval=10,
        )

        # Step 2: Run Real AI Processing
        update_job_state(
            state_file=state_file,
            status="PROCESSING",
            stage="AI_PERCEPTION",
            stage_label="Executing Vehicle Detection, ByteTrack, Plate OCR & Re-ID...",
            progress_percent=40,
        )

        telemetry = pipeline.process_video(
            video_source=str(video_path),
            output_video_path=str(raw_annotated_video),
            output_json_path=str(telemetry_json),
            max_frames=max_frames,
            save_video=True,
        )

        summary_stats = telemetry.get("summary_statistics", {})
        vehicle_tracks = telemetry.get("vehicle_tracks", [])

        # Step 3: Web Video Transcoding
        update_job_state(
            state_file=state_file,
            status="PROCESSING",
            stage="VIDEO_TRANSCODING",
            stage_label="Optimizing annotated video for instant HTML5 playback...",
            progress_percent=75,
        )

        if raw_annotated_video.exists():
            transcode_to_web_mp4(raw_annotated_video, web_annotated_video)
        else:
            logger.warning(f"Raw annotated video not found at {raw_annotated_video}")

        # Step 4: Database Ingestion via existing ingest_service
        update_job_state(
            state_file=state_file,
            status="PROCESSING",
            stage="DATABASE_INGESTION",
            stage_label=f"Persisting {len(vehicle_tracks)} vehicle observations to PostgreSQL/PostGIS...",
            progress_percent=85,
        )

        db_ingest_count = 0
        try:
            from app.db.session import SessionLocal
            from app.schemas.ai_contracts import VehicleTrackInput
            from app.services.ingest_service import ingest_tracks

            db = SessionLocal()
            try:
                # Convert raw track dicts to validated VehicleTrackInput objects
                validated_tracks = []
                for tr in vehicle_tracks:
                    try:
                        validated_tracks.append(VehicleTrackInput.model_validate(tr))
                    except Exception as val_err:
                        logger.warning(f"Skipping track {tr.get('track_id')} validation: {val_err}")

                if validated_tracks:
                    db_ingest_count = ingest_tracks(
                        db=db,
                        camera_id=camera_id,
                        tracks=validated_tracks,
                        commit=True,
                    )
                    logger.info(f"Successfully ingested {db_ingest_count} observations into DB for {camera_id}.")
            finally:
                db.close()
        except Exception as db_err:
            logger.error(f"Database ingestion error: {db_err}", exc_info=True)
            # We record error notice in metrics but allow pipeline completion
            summary_stats["db_ingest_error"] = str(db_err)

        summary_stats["db_ingested_observations"] = db_ingest_count

        # Step 5: Completed
        metrics_payload = {
            "total_frames": telemetry.get("metadata", {}).get("total_frames", 0),
            "processed_frames": telemetry.get("metadata", {}).get("processed_frames", 0),
            "pipeline_fps": telemetry.get("metadata", {}).get("pipeline_fps", 0.0),
            "processing_time_sec": telemetry.get("metadata", {}).get("total_processing_time_sec", 0.0),
            "vehicles_detected": summary_stats.get("total_vehicle_tracks", len(vehicle_tracks)),
            "vehicle_tracks": summary_stats.get("total_vehicle_tracks", len(vehicle_tracks)),
            "plate_detections": summary_stats.get("total_plate_detections", 0),
            "plate_associations": summary_stats.get("total_plate_associations", 0),
            "ocr_attempts": summary_stats.get("total_ocr_attempts", 0),
            "successful_ocr_reads": summary_stats.get("successful_ocr_reads", 0),
            "stable_plate_tracks": summary_stats.get("stable_plate_tracks", 0),
            "tentative_plate_tracks": summary_stats.get("tentative_plate_tracks", 0),
            "reid_embeddings": summary_stats.get("total_reid_observations", 0),
            "db_ingested_observations": db_ingest_count,
            "device": telemetry.get("metadata", {}).get("device", "cpu"),
            "detections": [
                {
                    "track_id": t.get("track_id"),
                    "vehicle_class": t.get("vehicle_class"),
                    "plate_text": t.get("plate", {}).get("text") if isinstance(t.get("plate"), dict) else t.get("plate_text"),
                    "plate_status": t.get("plate", {}).get("status") if isinstance(t.get("plate"), dict) else t.get("plate_status"),
                    "confidence": t.get("plate", {}).get("confidence") if isinstance(t.get("plate"), dict) else t.get("plate_confidence"),
                    "frame_count": t.get("frame_count", 0),
                    "reid_count": t.get("reid", {}).get("observation_count", 0) if isinstance(t.get("reid"), dict) else 0,
                }
                for t in vehicle_tracks[:50]  # Cap summary array to first 50 for fast JSON transmission
            ],
        }

        video_url = f"/api/v1/video-testing/jobs/{job_id}/video"

        update_job_state(
            state_file=state_file,
            status="COMPLETED",
            stage="COMPLETED",
            stage_label="AI video processing and database ingestion complete.",
            progress_percent=100,
            metrics=metrics_payload,
            video_relative_url=video_url,
        )
        logger.info(f"Job {job_id} successfully completed.")

    except Exception as e:
        logger.error(f"Pipeline failed for job {job_id}: {e}", exc_info=True)
        update_job_state(
            state_file=state_file,
            status="FAILED",
            stage="FAILED",
            stage_label="AI processing failed.",
            progress_percent=100,
            error=str(e),
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run NETRA Live Video Testing AI Pipeline")
    parser.add_argument("--job-id", required=True, help="Unique processing job UUID")
    parser.add_argument("--camera-id", required=True, help="Selected Camera ID (e.g. CAM_01)")
    parser.add_argument("--video-path", required=True, help="Path to input test video")
    parser.add_argument("--output-dir", required=True, help="Output directory for job results")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames to process")
    args = parser.parse_args()

    video_file = Path(args.video_path).resolve()
    output_directory = Path(args.output_dir).resolve()

    run_pipeline(
        job_id=args.job_id,
        camera_id=args.camera_id,
        video_path=video_file,
        output_dir=output_directory,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
