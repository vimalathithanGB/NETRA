"""Automated integration tests for REAL AI -> Backend -> PostgreSQL/PostGIS integration."""
import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text, select

from app.main import app
from app.db.session import engine, SessionLocal
from app.schemas.ai_contracts import (
    VehicleTrackInput,
    CrossCameraMatchInput,
    GlobalVehicleInput,
    TrajectoryInput,
    RoutePredictionInput,
)
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

client = TestClient(app)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real_ai_outputs"


@pytest.fixture(scope="module")
def real_ai_payloads():
    """Load the authoritative real AI output files."""
    with open(FIXTURES_DIR / "track_output.json") as f:
        track = json.load(f)
    with open(FIXTURES_DIR / "cross_camera_match.json") as f:
        match = json.load(f)
    with open(FIXTURES_DIR / "global_vehicle.json") as f:
        gv = json.load(f)
    with open(FIXTURES_DIR / "trajectory.json") as f:
        traj = json.load(f)
    with open(FIXTURES_DIR / "route_prediction.json") as f:
        pred = json.load(f)

    return {
        "track": track,
        "match": match,
        "global_vehicle": gv,
        "trajectory": traj,
        "prediction": pred,
    }


def test_1_real_ai_payload_validation(real_ai_payloads):
    """1. Verify real AI payloads validate strictly against Pydantic schemas."""
    # Track
    t = VehicleTrackInput(**real_ai_payloads["track"])
    assert t.track_id == 39
    assert t.plate_text == "ONDUTY"
    assert t.plate_status == "tentative"
    assert t.plate_confidence == 0.9681

    # Match (nullable plate_match, float class_match)
    m = CrossCameraMatchInput(**real_ai_payloads["match"])
    assert m.camera_a == "CAM_01"
    assert m.plate_match is None
    assert m.class_match == 1.0
    assert m.reid_similarity == 0.9611
    assert m.matched is True

    # Global vehicle
    gv = GlobalVehicleInput(**real_ai_payloads["global_vehicle"])
    assert gv.global_vehicle_id == "GV_000001"
    assert len(gv.observations) == 3

    # Trajectory & segments (segments have global_vehicle_id)
    traj = TrajectoryInput(**real_ai_payloads["trajectory"])
    assert traj.global_vehicle_id == "GV_000001"
    assert len(traj.segments) == 1
    assert traj.segments[0].global_vehicle_id == "GV_000001"
    assert traj.segments[0].travel_time_feasible is False
    assert traj.segments[0].direction == "NE"

    # Route prediction
    pred = RoutePredictionInput(**real_ai_payloads["prediction"])
    assert pred.global_vehicle_id == "GV_000002"
    assert len(pred.predictions) == 2


def test_2_successful_ingestion_individual_endpoints(real_ai_payloads):
    """2. Ingest each real AI artifact through individual FastAPI endpoints."""
    # Tracks
    res_track = client.post(
        "/api/v1/ingest/tracks",
        json={"camera_id": "CAM_01", "tracks": [real_ai_payloads["track"]]},
    )
    assert res_track.status_code == 200
    assert res_track.json()["success"] is True
    assert res_track.json()["counts"]["tracks"] == 1

    # Cross-camera matches (single object or list)
    res_match = client.post("/api/v1/ingest/matches", json=real_ai_payloads["match"])
    assert res_match.status_code == 200
    assert res_match.json()["success"] is True
    assert res_match.json()["counts"]["matches"] == 1

    # Global vehicles
    res_gv = client.post("/api/v1/ingest/global-vehicles", json=real_ai_payloads["global_vehicle"])
    assert res_gv.status_code == 200
    assert res_gv.json()["success"] is True
    assert res_gv.json()["counts"]["global_vehicles"] == 1

    # Trajectories
    res_traj = client.post("/api/v1/ingest/trajectories", json=real_ai_payloads["trajectory"])
    assert res_traj.status_code == 200
    assert res_traj.json()["success"] is True
    assert res_traj.json()["counts"]["trajectories"] == 1

    # Route predictions
    res_pred = client.post("/api/v1/ingest/predictions", json=real_ai_payloads["prediction"])
    assert res_pred.status_code == 200
    assert res_pred.json()["success"] is True
    assert res_pred.json()["counts"]["predictions"] == 1


def test_3_successful_ingestion_unified_endpoint(real_ai_payloads):
    """3. Ingest complete real AI payload composite through POST /api/v1/ingest."""
    unified_payload = {
        "camera_id": "CAM_01",
        "tracks": [real_ai_payloads["track"]],
        "matches": [real_ai_payloads["match"]],
        "global_vehicles": [real_ai_payloads["global_vehicle"]],
        "trajectories": [real_ai_payloads["trajectory"]],
        "predictions": [real_ai_payloads["prediction"]],
    }
    response = client.post("/api/v1/ingest", json=unified_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["counts"]["tracks"] == 1
    assert data["counts"]["matches"] == 1
    assert data["counts"]["global_vehicles"] == 1
    assert data["counts"]["trajectories"] == 1
    assert data["counts"]["predictions"] == 1


def test_4_database_persistence_and_entity_inspection():
    """4. Connect directly to netra_db and verify persistence in domain tables."""
    db = SessionLocal()
    try:
        # Check Camera
        cam = db.get(Camera, "CAM_01")
        assert cam is not None
        assert cam.status == "online"

        # Check GlobalVehicle
        gv = db.get(GlobalVehicle, "GV_000001")
        assert gv is not None
        assert gv.vehicle_class == "bus"
        assert gv.match_confidence == 0.9082

        # Check VehicleObservation
        obs = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 39,
            )
        ).scalar_one_or_none()
        assert obs is not None
        assert obs.plate_text == "ONDUTY"
        assert obs.plate_status == "tentative"
        assert obs.plate_confidence == 0.9681
        assert obs.observation_count == 5
        assert obs.valid_observation_count == 0

        # Check LicensePlate
        plate = db.execute(
            select(LicensePlate).where(LicensePlate.plate_text == "ONDUTY")
        ).scalar_one_or_none()
        assert plate is not None
        assert plate.plate_status == "tentative"
        assert plate.confidence in (0.9681, 1.0)

        # Check CrossCameraMatch
        match = db.execute(
            select(CrossCameraMatch).where(
                CrossCameraMatch.camera_a == "CAM_01",
                CrossCameraMatch.camera_b == "CAM_02",
            )
        ).first()
        assert match is not None

        # Check VehicleTrajectory & TrajectorySegment
        traj = db.execute(
            select(VehicleTrajectory).where(VehicleTrajectory.global_vehicle_id == "GV_000001")
        ).scalar_one_or_none()
        assert traj is not None
        assert traj.camera_sequence == ["CAM_01", "CAM_02", "CAM_03"]
        assert traj.total_distance_meters == 850.98

        seg = db.execute(
            select(TrajectorySegment).where(TrajectorySegment.trajectory_id == traj.id)
        ).scalar_one_or_none()
        assert seg is not None
        assert seg.from_camera == "CAM_01"
        assert seg.to_camera == "CAM_02"
        assert seg.travel_time_feasible is False
        assert seg.direction == "NE"

        # Check RoutePrediction
        pred = db.execute(
            select(RoutePrediction).where(RoutePrediction.global_vehicle_id == "GV_000002")
        ).first()
        assert pred is not None
    finally:
        db.close()


def test_5_relational_integrity():
    """5. Verify foreign key links across the domain model graph."""
    with engine.connect() as conn:
        # Verify GV_000001 joins to Trajectory and TrajectorySegment
        row = conn.execute(
            text(
                """
            SELECT 
                gv.global_vehicle_id,
                gv.vehicle_class,
                t.id as trajectory_id,
                ts.from_camera,
                ts.to_camera
            FROM global_vehicles gv
            JOIN vehicle_trajectories t ON gv.global_vehicle_id = t.global_vehicle_id
            JOIN trajectory_segments ts ON t.id = ts.trajectory_id
            WHERE gv.global_vehicle_id = 'GV_000001'
        """
            )
        ).fetchone()

        assert row is not None
        assert row.global_vehicle_id == "GV_000001"
        assert row.from_camera == "CAM_01"
        assert row.to_camera == "CAM_02"

        # Verify linked observations for GV_000001
        obs_rows = conn.execute(
            text(
                """
            SELECT camera_id, track_id
            FROM vehicle_observations
            WHERE global_vehicle_id = 'GV_000001'
            ORDER BY camera_id, track_id
        """
            )
        ).fetchall()

        cameras_tracks = [(r.camera_id, r.track_id) for r in obs_rows]
        assert ("CAM_01", 1) in cameras_tracks
        assert ("CAM_02", 2) in cameras_tracks
        assert ("CAM_03", 2) in cameras_tracks


def test_6_frontend_read_api_queries():
    """6. Read back inserted data through backend read APIs."""
    # Cameras
    r_cam = client.get("/api/v1/cameras")
    assert r_cam.status_code == 200
    cam_ids = [c["id"] for c in r_cam.json()]
    assert "CAM_01" in cam_ids
    assert "CAM_02" in cam_ids
    assert "CAM_03" in cam_ids

    # Trajectory by global_vehicle_id
    r_traj = client.get("/api/v1/vehicles/GV_000001/trajectory")
    assert r_traj.status_code == 200
    data_traj = r_traj.json()
    assert data_traj["status"] == "Active Track"
    assert len(data_traj["detections"]) >= 2
    assert data_traj["detections"][0]["cameraId"] == "CAM_01"
    assert data_traj["detections"][1]["cameraId"] == "CAM_02"

    # Compatibility route /api/vehicles/{id}/trajectory
    r_compat = client.get("/api/vehicles/GV_000001/trajectory")
    assert r_compat.status_code == 200
    assert r_compat.json()["plate"] == "GV_000001"

    # Plate captures
    r_plates = client.get("/api/v1/plates/captures")
    assert r_plates.status_code == 200
    onduty = [p for p in r_plates.json() if p["plate"] == "ONDUTY"]
    assert len(onduty) >= 1
    assert onduty[0]["camId"] in ("CAM_01", "CAM_02")
    assert onduty[0]["status"] == "Review"  # Tentative plate mapped to Review


def test_7_invalid_payload_rejection():
    """7. Verify Pydantic extra='forbid' strictly rejects fabricated fields."""
    bad_payload = {
        "track_id": 999,
        "vehicle_class": "Car",
        "fabricated_speed": 120.5,  # FABRICATED FIELD
    }
    response = client.post(
        "/api/v1/ingest/tracks",
        json={"camera_id": "CAM_01", "tracks": [bad_payload]},
    )
    assert response.status_code == 422  # Unprocessable Entity from Pydantic
