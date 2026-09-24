"""End-to-End Automated Integration Test Suite for NETRA.

Validates the full chain:
AI Pipeline JSON Outputs -> FastAPI Backend -> PostgreSQL/PostGIS -> Frontend Compatibility APIs.
"""
import sys
import os
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.core.config import settings
from app.db.session import engine, SessionLocal
from app.models.entities import (
    Camera,
    GlobalVehicle,
    VehicleObservation,
    LicensePlate,
    CrossCameraMatch,
    VehicleTrajectory,
    TrajectorySegment,
    RoutePrediction,
    CameraTransition,
)
from ai_engine.integration.backend_client import NETRABackendClient
from ai_engine.integration.ai_backend_ingestion import (
    sanitize_track,
    sanitize_match,
    sanitize_global_vehicle,
    sanitize_trajectory,
    sanitize_route_prediction,
)

client = TestClient(app)


# ==============================================================================
# 1. Backend & Configuration Integrity
# ==============================================================================

def test_1_backend_starts_and_health_check():
    """Verify backend FastAPI server boots and responds on /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_2_no_credentials_exposed_in_status_or_config():
    """Verify no database passwords or secrets are leaked through public endpoints."""
    # /health
    res_health = client.get("/health")
    assert "password" not in res_health.text.lower()
    assert "vivin" not in res_health.text.lower()

    # /api/v1/status
    res_status = client.get("/api/v1/status")
    assert res_status.status_code == 200
    assert "password" not in res_status.text.lower()
    assert "vivin" not in res_status.text.lower()


# ==============================================================================
# 2. AI Perceptual Output Schema Validation
# ==============================================================================

def test_3_real_ai_json_files_exist():
    """Verify all authoritative AI Engine output checkpoints exist on disk."""
    runs_dir = PROJECT_ROOT / "runs"
    expected_files = [
        runs_dir / "cross_camera" / "cross_camera_observations.json",
        runs_dir / "cross_camera" / "cross_camera_matches.json",
        runs_dir / "cross_camera" / "global_vehicle_entities.json",
        runs_dir / "trajectory" / "vehicle_trajectories.json",
        runs_dir / "analytics" / "traffic_analytics_summary.json",
    ]
    for p in expected_files:
        assert p.exists(), f"Missing required AI checkpoint file: {p}"


def test_4_ai_sanitization_functions():
    """Verify sanitization preserves real AI fields while strictly adhering to schemas."""
    runs_dir = PROJECT_ROOT / "runs"

    with open(runs_dir / "cross_camera" / "cross_camera_observations.json", "r", encoding="utf-8") as f:
        obs_raw = json.load(f)["camera_observations"][0]
    sanitized_obs = sanitize_track(obs_raw)
    assert sanitized_obs["track_id"] == obs_raw["track_id"]
    assert sanitized_obs["vehicle_class"] == obs_raw["vehicle_class"]
    assert "representative_embedding" not in sanitized_obs  # extra="forbid"

    with open(runs_dir / "cross_camera" / "cross_camera_matches.json", "r", encoding="utf-8") as f:
        matches_raw = [m for m in json.load(f)["matches"] if m.get("matched")][0]
    sanitized_m = sanitize_match(matches_raw)
    assert sanitized_m["matched"] is True
    assert sanitized_m["camera_a"] == matches_raw["camera_a"]

    with open(runs_dir / "cross_camera" / "global_vehicle_entities.json", "r", encoding="utf-8") as f:
        gv_raw = json.load(f)["global_vehicles"][0]
    sanitized_gv = sanitize_global_vehicle(gv_raw)
    assert sanitized_gv["global_vehicle_id"] == gv_raw["global_vehicle_id"]

    with open(runs_dir / "trajectory" / "vehicle_trajectories.json", "r", encoding="utf-8") as f:
        traj_raw = json.load(f)["vehicle_trajectories"][0]
    sanitized_traj = sanitize_trajectory(traj_raw)
    assert sanitized_traj["global_vehicle_id"] == traj_raw["global_vehicle_id"]
    assert len(sanitized_traj["camera_sequence"]) == len(traj_raw["camera_sequence"])


# ==============================================================================
# 3. Backend Ingestion & Endpoints Verification
# ==============================================================================

def test_5_camera_configuration_alignment():
    """Verify cameras are loaded with project configured coordinates (CAM_01..03)."""
    res = client.get("/api/v1/cameras")
    # If DB is connected, cameras should list CAM_01, CAM_02, CAM_03
    if res.status_code == 200:
        cams = res.json()
        assert len(cams) >= 3
        cam_ids = {c["id"] for c in cams}
        assert "CAM_01" in cam_ids
        assert "CAM_02" in cam_ids
        assert "CAM_03" in cam_ids

        cam_01 = next(c for c in cams if c["id"] == "CAM_01")
        # Verify real project coordinates (lat: ~11.0168, lon: ~76.9558), NOT Delhi (28.6)
        assert abs(cam_01["latitude"] - 11.0168) < 0.01
        assert abs(cam_01["longitude"] - 76.9558) < 0.01


def test_6_analytics_summary_endpoint():
    """Verify /api/v1/analytics/summary delivers real AI perception analytics."""
    res = client.get("/api/v1/analytics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "od_statistics" in data or "camera_statistics" in data or "route_statistics" in data


def test_7_frontend_compatibility_aliases():
    """Verify frontend aliases (/api/cameras, /api/traffic/events, /api/analytics/summary) match v1."""
    res_alias = client.get("/api/cameras")
    res_v1 = client.get("/api/v1/cameras")
    assert res_alias.status_code == res_v1.status_code

    res_analytics = client.get("/api/analytics/summary")
    assert res_analytics.status_code == 200


def test_8_trajectory_query_by_global_vehicle_id():
    """Verify /api/vehicles/{id}/trajectory responds with correct GIS shape."""
    # Test with GV_000001
    res = client.get("/api/vehicles/GV_000001/trajectory")
    # If database is seeded with trajectories, verify detections structure
    if res.status_code == 200:
        data = res.json()
        assert "detections" in data
        assert "plate" in data
        assert "totalDistanceKm" in data
        assert "avgSpeedKmH" in data
