"""Integration tests for Live Video Testing API endpoints."""
import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import SessionLocal
from app.models.entities import Camera


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_camera():
    db = SessionLocal()
    cam = db.query(Camera).first()
    if not cam:
        cam = Camera(
            id="CAM_TEST_01",
            name="Test Verification Camera",
            latitude=11.0168,
            longitude=76.9558,
            status="online",
            fps=30,
        )
        db.add(cam)
        db.commit()
        db.refresh(cam)
    cam_id = cam.id
    db.close()
    return cam_id


def test_get_nonexistent_job_status(client):
    """Verify 404 on nonexistent job."""
    resp = client.get("/api/v1/video-testing/jobs/NONEXISTENT-XYZ-9999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_upload_invalid_camera(client):
    """Verify 404 when uploading for a camera that does not exist."""
    fake_video = io.BytesIO(b"fake video content")
    resp = client.post(
        "/api/v1/video-testing/jobs",
        data={"camera_id": "DEFINITELY_NOT_A_REAL_CAMERA_12345"},
        files={"file": ("test.mp4", fake_video, "video/mp4")},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_upload_invalid_file_extension(client, sample_camera):
    """Verify 400 when uploading an unsupported file format."""
    fake_txt = io.BytesIO(b"this is a text file")
    resp = client.post(
        "/api/v1/video-testing/jobs",
        data={"camera_id": sample_camera},
        files={"file": ("test.txt", fake_txt, "text/plain")},
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


def test_upload_valid_video_creates_job(client, sample_camera):
    """Verify 201 Created and processing status on valid video upload."""
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    video_path = project_root / "data" / "videos" / "test.mp4"

    with open(video_path, "rb") as vf:
        resp = client.post(
            "/api/v1/video-testing/jobs",
            data={"camera_id": sample_camera},
            files={"file": ("test.mp4", vf, "video/mp4")},
        )
    assert resp.status_code == 201
    data = resp.json()
    assert "job_id" in data
    assert data["camera_id"] == sample_camera
    assert data["status"] == "QUEUED"
    assert "message" in data

    # Poll status immediately
    job_id = data["job_id"]
    status_resp = client.get(f"/api/v1/video-testing/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["camera_id"] == sample_camera
    assert status_data["status"] in ("QUEUED", "PROCESSING", "COMPLETED")
    assert "stage" in status_data


def test_compat_route_alias(client):
    """Verify /api/video-testing compatibility alias exists and responds."""
    resp = client.get("/api/video-testing/jobs/NONEXISTENT-XYZ-9999")
    assert resp.status_code == 404
