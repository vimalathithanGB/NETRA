from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify that GET /health returns HTTP 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_v1_status():
    """Verify that GET /api/v1/status returns API status information."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == "v1"
    assert data["status"] == "ready"
    assert "detections" in data["modules"]
    assert data["modules"]["detections"] == "waiting_for_ai_json_contract"


def test_openapi_docs_accessible():
    """Verify that the OpenAPI Swagger documentation is reachable."""
    response = client.get("/docs")
    assert response.status_code == 200
