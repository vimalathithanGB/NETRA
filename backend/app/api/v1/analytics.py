"""Traffic Intelligence and Analytics API Routes.

Exposes verified AI-generated traffic flow, OD matrix, transition probabilities,
density heatmaps, and vehicle class statistics for NETRA Dashboard.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.db.session import get_db
from app.models.entities import (
    Camera,
    GlobalVehicle,
    VehicleObservation,
    CameraTransition,
    VehicleTrajectory,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])

# Global memory cache for live ingested analytics
_analytics_cache: Dict[str, Any] = {}


def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def _load_analytics_checkpoint() -> Optional[Dict[str, Any]]:
    """Try loading the real AI analytics checkpoint from runs/ directory."""
    runs_dir = _get_project_root() / "runs" / "analytics"
    summary_file = runs_dir / "traffic_analytics_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading analytics summary file: {e}")
    return None


@router.get("/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    """
    Retrieve comprehensive traffic analytics summary.
    Prioritizes memory cache -> runs/ checkpoint -> live database aggregation.
    """
    if _analytics_cache:
        return _analytics_cache

    disk_summary = _load_analytics_checkpoint()
    if disk_summary:
        return disk_summary

    # Fallback to database dynamic aggregation
    total_gv = db.scalar(select(func.count(GlobalVehicle.global_vehicle_id))) or 0
    total_obs = db.scalar(select(func.count(VehicleObservation.id))) or 0
    total_traj = db.scalar(select(func.count(VehicleTrajectory.id))) or 0
    cameras = db.execute(select(Camera)).scalars().all()

    camera_stats = {}
    heatmap_stats = []
    for cam in cameras:
        obs_count = db.scalar(
            select(func.count(VehicleObservation.id)).where(VehicleObservation.camera_id == cam.id)
        ) or 0
        camera_stats[cam.id] = {
            "camera_id": cam.id,
            "total_observations": obs_count,
            "name": cam.name,
            "latitude": cam.latitude,
            "longitude": cam.longitude,
        }
        heatmap_stats.append({
            "camera_id": cam.id,
            "latitude": cam.latitude,
            "longitude": cam.longitude,
            "road_name": cam.name,
            "vehicle_count": obs_count,
            "traffic_intensity": min(1.0, obs_count / 50.0) if obs_count else 0.0,
        })

    transitions = db.execute(select(CameraTransition)).scalars().all()
    route_densities = [
        {
            "from_camera": t.from_camera,
            "to_camera": t.to_camera,
            "vehicle_count": t.transition_count,
        }
        for t in transitions
    ]

    cams = [cam.id for cam in cameras]
    matrix = {c1: {c2: 0 for c2 in cams} for c1 in cams}
    origin_counts = {c: 0 for c in cams}
    dest_counts = {c: 0 for c in cams}
    total_trips = 0
    for t in transitions:
        if t.from_camera in matrix and t.to_camera in matrix[t.from_camera]:
            matrix[t.from_camera][t.to_camera] = t.transition_count
            origin_counts[t.from_camera] += t.transition_count
            dest_counts[t.to_camera] += t.transition_count
            total_trips += t.transition_count

    od_stats = {
        "cameras": cams,
        "matrix": matrix,
        "origin_counts": origin_counts,
        "destination_counts": dest_counts,
        "total_trips": total_trips,
    }

    return {
        "dataset_type": "database_aggregated",
        "total_global_vehicles": total_gv,
        "total_observations": total_obs,
        "total_trajectories": total_traj,
        "camera_statistics": camera_stats,
        "heatmap_statistics": heatmap_stats,
        "od_statistics": od_stats,
        "route_statistics": {
            "route_densities": route_densities,
            "total_transitions": sum(t.transition_count for t in transitions),
        },
    }


@router.post("/ingest", status_code=status.HTTP_200_OK)
def ingest_analytics_summary(payload: Dict[str, Any]):
    """Ingest new traffic analytics summary checkpoint from AI Engine."""
    global _analytics_cache
    _analytics_cache = payload
    logger.info("Updated live analytics memory cache from AI engine ingestion.")
    return {"success": True, "message": "Analytics summary ingested successfully"}


@router.get("/heatmap")
def get_traffic_heatmap(db: Session = Depends(get_db)):
    """Retrieve normalized spatial traffic observation heatmap points."""
    summary = get_analytics_summary(db)
    return {
        "heatmap_points": summary.get("heatmap_statistics", [])
    }


@router.get("/od-matrix")
def get_od_matrix(db: Session = Depends(get_db)):
    """Retrieve Origin-Destination flow matrix."""
    summary = get_analytics_summary(db)
    return summary.get("od_statistics", {})


@router.get("/routes")
def get_route_densities(db: Session = Depends(get_db)):
    """Retrieve corridor densities between camera nodes."""
    summary = get_analytics_summary(db)
    return summary.get("route_statistics", {})
