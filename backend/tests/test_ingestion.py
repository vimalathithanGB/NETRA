"""Tests for AI Engine ingestion endpoints using confirmed AI output contracts."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_ingest_camera_tracks_valid():
    """Verify ingestion of confirmed Vehicle Track JSON outputs."""
    payload = {
        "camera_id": "CAM-001",
        "tracks": [
            {
                "track_id": 101,
                "vehicle_class": "Car",
                "plate_text": "TN38AB1234",
                "plate_status": "stable",
                "plate_confidence": 99.4,
                "observation_count": 45,
                "valid_observation_count": 42,
                "best_ocr_confidence": 99.4,
                "first_seen_frame": 100,
                "last_seen_frame": 145,
            },
            {
                "track_id": 102,
                "vehicle_class": "Commercial SUV",
                "plate_text": "DL04C8891",
                "plate_status": "tentative",
                "plate_confidence": 88.5,
                "observation_count": 20,
                "valid_observation_count": 18,
                "best_ocr_confidence": 92.1,
                "first_seen_frame": 200,
                "last_seen_frame": 220,
            }
        ]
    }
    response = client.post("/api/v1/ingest/tracks", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["counts"]["tracks"] >= 1


def test_ingest_camera_tracks_invalid_extra_field():
    """Verify rejection when payload contains fabricated/unsupported fields."""
    payload = {
        "camera_id": "CAM-001",
        "tracks": [
            {
                "track_id": 103,
                "vehicle_class": "Car",
                "fabricated_confidence": 99.9,  # Unconfirmed field -> must be rejected
            }
        ]
    }
    response = client.post("/api/v1/ingest/tracks", json=payload)
    assert response.status_code == 422


def test_ingest_cross_camera_matches_valid():
    """Verify ingestion of confirmed Cross-Camera Match JSON."""
    payload = [
        {
            "camera_a": "CAM-001",
            "track_a": 101,
            "camera_b": "CAM-004",
            "track_b": 201,
            "reid_similarity": 0.89,
            "plate_match": True,
            "class_match": True,
            "available_evidence": ["reid", "plate"],
            "match_score": 0.94,
            "threshold": 0.85,
            "matched": True,
        }
    ]
    response = client.post("/api/v1/ingest/matches", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["counts"]["matches"] == 1


def test_ingest_global_vehicles_valid():
    """Verify ingestion of confirmed Global Vehicle JSON."""
    payload = [
        {
            "global_vehicle_id": "GV-TEST-001",
            "vehicle_class": "Car",
            "observations": [
                {"camera_id": "CAM-001", "track_id": 101},
                {"camera_id": "CAM-004", "track_id": 201},
            ],
            "plate_text": "TN38AB1234",
            "match_confidence": 98.5,
        }
    ]
    response = client.post("/api/v1/ingest/global-vehicles", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["counts"]["global_vehicles"] == 1


def test_ingest_trajectories_valid():
    """Verify ingestion of confirmed Trajectory JSON."""
    payload = [
        {
            "global_vehicle_id": "GV-TEST-001",
            "vehicle_class": "Car",
            "plate_text": "TN38AB1234",
            "camera_sequence": ["CAM-001", "CAM-004"],
            "total_distance_meters": 5500.0,
            "total_travel_time_seconds": 272.0,
            "average_speed_kmh": 58.0,
            "segments": [
                {
                    "from_camera": "CAM-001",
                    "to_camera": "CAM-004",
                    "from_timestamp": "14:32:10",
                    "to_timestamp": "14:36:42",
                    "distance_meters": 5500.0,
                    "cumulative_distance_meters": 5500.0,
                    "travel_time_seconds": 272.0,
                    "average_speed_kmh": 58.0,
                    "speed_limit_kmh": 60.0,
                    "minimum_feasible_time_seconds": 200.0,
                    "travel_time_feasible": True,
                    "bearing_degrees": 135.0,
                    "direction": "South-East",
                }
            ],
        }
    ]
    response = client.post("/api/v1/ingest/trajectories", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["counts"]["trajectories"] == 1


def test_ingest_route_predictions_valid():
    """Verify ingestion of confirmed Route Prediction JSON."""
    payload = [
        {
            "global_vehicle_id": "GV-TEST-001",
            "current_camera": "CAM-004",
            "vehicle_class": "Car",
            "prediction_available": True,
            "predictions": [
                {
                    "next_camera": "CAM-007",
                    "probability": 0.74,
                    "observed_historical_transitions": 182,
                },
                {
                    "next_camera": "CAM-012",
                    "probability": 0.26,
                    "observed_historical_transitions": 64,
                }
            ],
        }
    ]
    response = client.post("/api/v1/ingest/predictions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["counts"]["predictions"] == 1


def test_ingest_unified_payload_valid():
    """Verify unified endpoint accepting composite AI pipeline outputs."""
    payload = {
        "camera_id": "CAM-002",
        "tracks": [
            {
                "track_id": 501,
                "vehicle_class": "Sedan",
                "plate_text": "HR26DK5092",
                "plate_status": "stable",
                "plate_confidence": 98.9,
            }
        ]
    }
    response = client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["counts"]["tracks"] >= 1
