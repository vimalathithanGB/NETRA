"""Tests for Frontend read endpoints and compatibility routing."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_cameras_canonical_and_compat():
    """Verify cameras API returns expected shape for both /api/v1 and /api prefixes."""
    # Canonical /api/v1/cameras
    res_v1 = client.get("/api/v1/cameras")
    assert res_v1.status_code == 200
    cameras_v1 = res_v1.json()
    assert isinstance(cameras_v1, list)
    assert len(cameras_v1) >= 3

    # Verify first camera structure matches gisService.ts Camera interface
    cam = cameras_v1[0]
    assert "id" in cam
    assert "name" in cam
    assert "latitude" in cam
    assert "longitude" in cam
    assert "status" in cam
    assert isinstance(cam["latitude"], float)
    assert isinstance(cam["longitude"], float)

    # Compatibility bridge /api/cameras
    res_compat = client.get("/api/cameras")
    assert res_compat.status_code == 200
    assert res_compat.json() == cameras_v1


def test_get_vehicle_trajectory_canonical_and_compat():
    """Verify trajectory retrieval for preset vehicle."""
    plate = "TN38AB1234"

    # Canonical /api/v1/vehicles/{plate}/trajectory
    res_v1 = client.get(f"/api/v1/vehicles/{plate}/trajectory")
    assert res_v1.status_code == 200
    data_v1 = res_v1.json()
    assert data_v1["plate"] == plate
    assert "detections" in data_v1
    assert len(data_v1["detections"]) >= 2
    assert "totalDistanceKm" in data_v1
    assert "avgSpeedKmH" in data_v1

    # Verify first detection item matches VehicleDetection interface
    det = data_v1["detections"][0]
    assert "cameraId" in det
    assert "cameraName" in det
    assert "latitude" in det
    assert "longitude" in det
    assert "timestamp" in det
    assert "plate" in det
    assert "vehicleType" in det

    # Compatibility bridge /api/vehicles/{plate}/trajectory
    res_compat = client.get(f"/api/vehicles/{plate}/trajectory")
    assert res_compat.status_code == 200
    assert res_compat.json() == data_v1


def test_get_unknown_vehicle_trajectory():
    """Verify 404 response for unknown vehicle."""
    res = client.get("/api/v1/vehicles/UNKNOWN9999/trajectory")
    assert res.status_code == 404


def test_get_traffic_events_canonical_and_compat():
    """Verify traffic zones and incidents telemetry."""
    # Canonical /api/v1/traffic/events
    res_v1 = client.get("/api/v1/traffic/events")
    assert res_v1.status_code == 200
    data_v1 = res_v1.json()
    assert "zones" in data_v1
    assert "incidents" in data_v1
    assert len(data_v1["zones"]) >= 4
    assert len(data_v1["incidents"]) >= 2

    # Verify zone structure
    z = data_v1["zones"][0]
    assert "id" in z
    assert "status" in z
    assert "center" in z
    assert "radiusMeters" in z

    # Compatibility bridge /api/traffic/events
    res_compat = client.get("/api/traffic/events")
    assert res_compat.status_code == 200
    assert res_compat.json() == data_v1


def test_get_plate_captures():
    """Verify ANPR plate captures endpoint."""
    res = client.get("/api/v1/plates/captures")
    assert res.status_code == 200
    captures = res.json()
    assert isinstance(captures, list)
    if captures:
        c = captures[0]
        assert "plate" in c
        assert "type" in c
        assert "camId" in c
        assert "timestamp" in c
        assert "confidence" in c
        assert "status" in c


def test_get_alerts():
    """Verify active violations and enforcement alerts endpoint."""
    res = client.get("/api/v1/alerts")
    assert res.status_code == 200
    alerts = res.json()
    assert isinstance(alerts, list)
    assert len(alerts) >= 4
    a = alerts[0]
    assert "id" in a
    assert "type" in a
    assert "plate" in a
    assert "location" in a
    assert "severity" in a
    assert "status" in a
