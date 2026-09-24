"""CP9 Comprehensive Test Suite: Database Schema & Ingestion Alignment.

Validates:
1. VehicleObservation new fields: last_bbox, detector_confidence
2. TrajectorySegment new fields: match_score, reid_similarity, available_evidence
3. Camera new fields: road_name, speed_limit_kmh
4. Alert table persistence & API operations
5. TrafficEvent table persistence & API operations
6. Ingestion data preservation from real Phase 1 manifests
7. Alembic migration head alignment
"""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from app.main import app
from app.db.session import engine, SessionLocal
from app.models.entities import (
    Camera,
    GlobalVehicle,
    VehicleObservation,
    LicensePlate,
    CrossCameraMatch,
    VehicleTrajectory,
    TrajectorySegment,
    Alert,
    TrafficEvent,
)
from app.schemas.ai_contracts import VehicleTrackInput, TrajectorySegmentInput
from app.services.camera_service import seed_default_cameras

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_cp9_alembic_migration_head():
    """Verify Alembic migration head is at CP9 revision."""
    with engine.connect() as conn:
        res = conn.execute(text("SELECT version_num FROM alembic_version;")).scalar()
        assert res == "2a4b6c8d1e3f"


def test_cp9_camera_model_alignment():
    """Verify Camera model has road_name and speed_limit_kmh and seeds accurately."""
    db = SessionLocal()
    try:
        seed_default_cameras(db)
        cam = db.get(Camera, "CAM_01")
        assert cam is not None
        assert cam.road_name == "SIMULATED_ROAD_A"
        assert cam.speed_limit_kmh == 40.0

        # Verify API response contains new fields
        res = client.get("/api/v1/cameras/CAM_01")
        assert res.status_code == 200
        data = res.json()
        assert data["roadName"] == "SIMULATED_ROAD_A"
        assert data["speedLimitKmh"] == 40.0
    finally:
        db.close()


def test_cp9_vehicle_observation_new_fields():
    """Verify VehicleObservation correctly ingests and stores last_bbox and detector_confidence."""
    db = SessionLocal()
    try:
        # Ingest track with last_bbox and detector_confidence
        payload = {
            "camera_id": "CAM_01",
            "tracks": [
                {
                    "track_id": 9901,
                    "vehicle_class": "bus",
                    "plate_text": "DL01AB9999",
                    "plate_status": "stable",
                    "plate_confidence": 98.5,
                    "observation_count": 25,
                    "valid_observation_count": 22,
                    "best_ocr_confidence": 98.5,
                    "last_bbox": [100, 200, 300, 400],
                    "detector_confidence": 0.8954,
                    "first_seen_frame": 10,
                    "last_seen_frame": 35,
                }
            ],
        }
        res = client.post("/api/v1/ingest/tracks", json=payload)
        assert res.status_code == 200
        assert res.json()["success"] is True

        obs = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 9901,
            )
        ).scalar_one_or_none()

        assert obs is not None
        assert obs.last_bbox == [100, 200, 300, 400]
        assert obs.detector_confidence == pytest.approx(0.8954, abs=1e-4)
    finally:
        # Cleanup
        to_del = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 9901,
            )
        ).scalars().all()
        for o in to_del:
            db.delete(o)
        db.commit()
        db.close()


def test_cp9_trajectory_segment_new_fields():
    """Verify TrajectorySegment ingests and stores match_score, reid_similarity, and available_evidence."""
    db = SessionLocal()
    try:
        payload = [
            {
                "global_vehicle_id": "GV_CP9_TEST",
                "vehicle_class": "bus",
                "plate_text": "HR26AB1111",
                "camera_sequence": ["CAM_01", "CAM_02"],
                "total_distance_meters": 366.38,
                "total_travel_time_seconds": 2.536,
                "average_speed_kmh": 520.13,
                "segments": [
                    {
                        "from_camera": "CAM_01",
                        "to_camera": "CAM_02",
                        "from_timestamp": "2026-01-01T09:30:00.033367",
                        "to_timestamp": "2026-01-01T09:30:02.569233",
                        "distance_meters": 366.38,
                        "cumulative_distance_meters": 366.38,
                        "travel_time_seconds": 2.536,
                        "average_speed_kmh": 520.13,
                        "speed_limit_kmh": 40.0,
                        "minimum_feasible_time_seconds": 32.974,
                        "travel_time_feasible": False,
                        "bearing_degrees": 43.25,
                        "direction": "NE",
                        "match_score": 0.9667,
                        "reid_similarity": 0.9611,
                        "available_evidence": ["reid", "class"],
                    }
                ],
            }
        ]
        res = client.post("/api/v1/ingest/trajectories", json=payload)
        assert res.status_code == 200
        assert res.json()["success"] is True

        traj = db.execute(
            select(VehicleTrajectory).where(VehicleTrajectory.global_vehicle_id == "GV_CP9_TEST")
        ).scalar_one_or_none()

        assert traj is not None
        assert len(traj.segments) == 1
        seg = traj.segments[0]
        assert seg.match_score == pytest.approx(0.9667, abs=1e-4)
        assert seg.reid_similarity == pytest.approx(0.9611, abs=1e-4)
        assert seg.available_evidence == ["reid", "class"]
    finally:
        # Cleanup
        to_del_traj = db.execute(
            select(VehicleTrajectory).where(VehicleTrajectory.global_vehicle_id == "GV_CP9_TEST")
        ).scalars().all()
        for t in to_del_traj:
            db.delete(t)
        to_del_gv = db.get(GlobalVehicle, "GV_CP9_TEST")
        if to_del_gv:
            db.delete(to_del_gv)
        db.commit()
        db.close()


def test_cp9_alert_persistence_and_api():
    """Verify Alert model persistence, API creation, and retrieval."""
    test_alert_id = "ALT-TEST-CP9-01"
    alert_payload = {
        "id": test_alert_id,
        "alert_type": "Overspeeding (>120 km/h)",
        "severity": "Critical",
        "message": "Vehicle exceeded corridor threshold by 40 km/h",
        "camera_id": "CAM_01",
        "plate_text": "UP16CP9999",
        "location_text": "Simulated Junction North",
        "timestamp_text": "14:45 IST",
        "status": "Pending Review",
    }
    # Register alert via API
    res_post = client.post("/api/v1/alerts", json=alert_payload)
    assert res_post.status_code == 201
    created = res_post.json()
    assert created["id"] == test_alert_id
    assert created["type"] == "Overspeeding (>120 km/h)"
    assert created["plate"] == "UP16CP9999"

    # Query alert via API
    res_get = client.get("/api/v1/alerts")
    assert res_get.status_code == 200
    all_alerts = res_get.json()
    assert any(a["id"] == test_alert_id for a in all_alerts)

    # Verify directly in PostgreSQL
    db = SessionLocal()
    try:
        record = db.get(Alert, test_alert_id)
        assert record is not None
        assert record.alert_type == "Overspeeding (>120 km/h)"
        assert record.severity == "Critical"
        assert record.plate_text == "UP16CP9999"
        assert record.camera_id == "CAM_01"
        assert record.status == "Pending Review"
    finally:
        # Cleanup
        to_del = db.get(Alert, test_alert_id)
        if to_del:
            db.delete(to_del)
            db.commit()
        db.close()


def test_cp9_traffic_event_persistence_and_api():
    """Verify TrafficEvent model persistence with PostGIS geometry and API operations."""
    test_event_id = "INC-TEST-CP9-99"
    incident_payload = {
        "id": test_event_id,
        "event_type": "incident",
        "title": "Lane Obstruction Test",
        "severity": "high",
        "camera_id": "CAM_02",
        "corridor": "Simulated Central Avenue",
        "latitude": 11.0192,
        "longitude": 76.9581,
        "description": "Debris in left lane, slow corridor transit",
        "timestamp_text": "14:50 IST",
        "status": "active",
    }
    # Register incident via API
    res_post = client.post("/api/v1/traffic/events/incidents", json=incident_payload)
    assert res_post.status_code == 201
    created = res_post.json()
    assert created["id"] == test_event_id
    assert created["title"] == "Lane Obstruction Test"

    # Query events via API
    res_get = client.get("/api/v1/traffic/events")
    assert res_get.status_code == 200
    data = res_get.json()
    assert "incidents" in data
    assert any(i["id"] == test_event_id for i in data["incidents"])

    # Verify directly in PostgreSQL with PostGIS
    db = SessionLocal()
    try:
        record = db.get(TrafficEvent, test_event_id)
        assert record is not None
        assert record.title == "Lane Obstruction Test"
        assert record.latitude == pytest.approx(11.0192, abs=1e-4)
        assert record.longitude == pytest.approx(76.9581, abs=1e-4)

        # Check PostGIS geometry representation
        with engine.connect() as conn:
            geom_text = conn.execute(
                text("SELECT ST_AsText(location_geom) FROM traffic_events WHERE id = :eid;"),
                {"eid": test_event_id},
            ).scalar()
            assert "POINT(76.9581 11.0192)" in geom_text
    finally:
        # Cleanup
        to_del = db.get(TrafficEvent, test_event_id)
        if to_del:
            db.delete(to_del)
            db.commit()
        db.close()


def test_cp9_real_phase1_manifest_data_preservation():
    """Verify ingesting real Phase 1 artifacts preserves all 10 CP7 fields in PostgreSQL."""
    obs_file = PROJECT_ROOT / "runs" / "cross_camera" / "cross_camera_observations.json"
    traj_file = PROJECT_ROOT / "runs" / "trajectory" / "vehicle_trajectories.json"

    assert obs_file.exists(), "cross_camera_observations.json must exist"
    assert traj_file.exists(), "vehicle_trajectories.json must exist"

    with open(obs_file, "r", encoding="utf-8") as f:
        obs_data = json.load(f)["camera_observations"]

    # Ingest CAM_01 observations
    cam01_obs = [o for o in obs_data if o["camera_id"] == "CAM_01"]
    formatted_tracks = [
        {
            "track_id": o["track_id"],
            "vehicle_class": o["vehicle_class"],
            "plate_text": o["plate_text"],
            "plate_status": o["plate_status"],
            "plate_confidence": o["plate_confidence"],
            "observation_count": o["observation_count"],
            "valid_observation_count": o["valid_observation_count"],
            "best_ocr_confidence": o["best_ocr_confidence"],
            "last_bbox": o["last_bbox"],
            "detector_confidence": o["detector_confidence"],
            "first_seen_frame": o["first_seen_frame"],
            "last_seen_frame": o["last_seen_frame"],
        }
        for o in cam01_obs
    ]

    res = client.post(
        "/api/v1/ingest/tracks",
        json={"camera_id": "CAM_01", "tracks": formatted_tracks},
    )
    assert res.status_code == 200

    # Ingest Trajectories
    with open(traj_file, "r", encoding="utf-8") as f:
        traj_data = json.load(f)["vehicle_trajectories"]

    res_traj = client.post("/api/v1/ingest/trajectories", json=traj_data)
    assert res_traj.status_code == 200

    # Connect to database and verify GV_000001 preservation
    db = SessionLocal()
    try:
        # Check CAM_01 track 1 last_bbox and detector_confidence
        obs1 = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 1,
            )
        ).scalar_one_or_none()

        assert obs1 is not None
        assert obs1.last_bbox == [696, 748, 1847, 1438]
        assert obs1.detector_confidence == pytest.approx(0.8769, abs=1e-4)

        # Check TrajectorySegment for GV_000001
        traj1 = db.execute(
            select(VehicleTrajectory).where(VehicleTrajectory.global_vehicle_id == "GV_000001")
        ).scalar_one_or_none()

        assert traj1 is not None
        assert len(traj1.segments) == 2
        seg1 = traj1.segments[0]
        assert seg1.from_camera == "CAM_01"
        assert seg1.to_camera == "CAM_02"
        assert seg1.match_score == pytest.approx(0.9667, abs=1e-4)
        assert seg1.reid_similarity == pytest.approx(0.9611, abs=1e-4)
        assert seg1.available_evidence == ["reid", "class"]
    finally:
        db.close()
