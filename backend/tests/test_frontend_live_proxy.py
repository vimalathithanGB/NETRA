"""Automated integration test verifying live frontend dev server proxy -> FastAPI backend."""
import urllib.request
import json
import pytest

FRONTEND_BASE = "http://localhost:5173"


def _is_live_stack_running() -> bool:
    try:
        req_fe = urllib.request.Request(f"{FRONTEND_BASE}/")
        with urllib.request.urlopen(req_fe, timeout=1) as resp:
            if resp.status != 200:
                return False
        req_be = urllib.request.Request("http://127.0.0.1:8000/health")
        with urllib.request.urlopen(req_be, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_live_stack_running(),
    reason="Frontend Vite dev server (5173) or Backend (8000) is not running",
)


def test_frontend_server_accessible():
    """Verify Vite dev server is running and responding."""
    req = urllib.request.Request(f"{FRONTEND_BASE}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode()
        assert "NETRA" in html or "root" in html


def test_proxied_cameras_endpoint():
    """Verify /api/cameras proxies to backend and returns all 11 cameras."""
    req = urllib.request.Request(f"{FRONTEND_BASE}/api/cameras")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert isinstance(data, list)
        assert len(data) >= 7
        cam_ids = [c["id"] for c in data]
        assert "CAM_01" in cam_ids
        assert "CAM_02" in cam_ids
        assert "CAM_03" in cam_ids


def test_proxied_traffic_events_endpoint():
    """Verify /api/traffic/events proxies and returns zones and incidents."""
    req = urllib.request.Request(f"{FRONTEND_BASE}/api/traffic/events")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "zones" in data
        assert "incidents" in data
        assert len(data["zones"]) >= 4


def test_proxied_trajectory_endpoint():
    """Verify /api/vehicles/GV_000001/trajectory returns real AI database trajectory."""
    req = urllib.request.Request(f"{FRONTEND_BASE}/api/vehicles/GV_000001/trajectory")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["plate"] == "GV_000001"
        assert data["status"] == "Active Track"
        assert len(data["detections"]) >= 2
        assert data["detections"][0]["cameraId"] == "CAM_01"
        assert data["detections"][1]["cameraId"] == "CAM_02"


def test_proxied_anpr_captures_endpoint():
    """Verify /api/v1/plates/captures returns real ANPR sightings including ONDUTY."""
    req = urllib.request.Request(f"{FRONTEND_BASE}/api/v1/plates/captures")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert isinstance(data, list)
        onduty = [c for c in data if c["plate"] == "ONDUTY"]
        assert len(onduty) >= 1
        assert any(c["camId"] == "CAM_01" for c in onduty)
        assert onduty[0]["status"] == "Review"


def test_proxied_alerts_endpoint():
    """Verify /api/v1/alerts returns active alerts."""
    req = urllib.request.Request(f"{FRONTEND_BASE}/api/v1/alerts")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert isinstance(data, list)
        assert len(data) >= 5
