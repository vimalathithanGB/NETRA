"""NETRA Backend Client for AI Engine Pipeline HTTP Ingestion.

Transfers verified AI Engine perception, Re-ID, cross-camera, and trajectory
outputs to the FastAPI backend via REST APIs.
"""
import os
import logging
from typing import Dict, Any, List, Optional
try:
    import httpx as http_lib
    _USE_HTTPX = True
except ImportError:
    try:
        import requests as http_lib
        _USE_HTTPX = False
    except ImportError:
        http_lib = None
        _USE_HTTPX = False

logger = logging.getLogger("NETRABackendClient")


class NETRABackendClient:
    """Client for transmitting AI Engine analytics to NETRA FastAPI backend."""

    def __init__(self, backend_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (
            backend_url
            or os.getenv("NETRA_BACKEND_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.timeout = timeout
        if _USE_HTTPX:
            self.client = http_lib.Client(
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "NETRA-AI-Engine-Ingest/1.0",
                }
            )
        elif http_lib is not None:
            self.client = http_lib.Session()
            self.client.headers.update({
                "Content-Type": "application/json",
                "User-Agent": "NETRA-AI-Engine-Ingest/1.0",
            })
        else:
            self.client = None

    def check_health(self) -> bool:
        """Check if FastAPI backend is operational."""
        if not self.client:
            return False
        try:
            resp = self.client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception as e:
            logger.warning(f"Backend health check failed: {e}")
            return False


    def ingest_tracks(self, camera_id: str, tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send vehicle observation tracks for a camera node."""
        endpoint = f"{self.base_url}/api/v1/ingest/tracks"
        payload = {"camera_id": camera_id, "tracks": tracks}
        return self._post(endpoint, payload)

    def ingest_matches(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send pairwise cross-camera Re-ID match results."""
        endpoint = f"{self.base_url}/api/v1/ingest/matches"
        return self._post(endpoint, matches)

    def ingest_global_vehicles(self, vehicles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send canonical global vehicle entities."""
        endpoint = f"{self.base_url}/api/v1/ingest/global-vehicles"
        return self._post(endpoint, vehicles)

    def ingest_trajectories(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send spatial corridors with segments."""
        endpoint = f"{self.base_url}/api/v1/ingest/trajectories"
        return self._post(endpoint, trajectories)

    def ingest_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send next-camera transition predictions."""
        endpoint = f"{self.base_url}/api/v1/ingest/predictions"
        return self._post(endpoint, predictions)

    def ingest_analytics(self, analytics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Send traffic analytics checkpoint summary."""
        endpoint = f"{self.base_url}/api/v1/analytics/ingest"
        return self._post(endpoint, analytics_summary)

    def ingest_unified(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a unified payload containing all components."""
        endpoint = f"{self.base_url}/api/v1/ingest"
        return self._post(endpoint, payload)

    def _post(self, endpoint: str, payload: Any) -> Dict[str, Any]:
        """Execute HTTP POST with error handling."""
        if not self.client:
            return {"success": False, "error": "No HTTP client available"}
        try:
            resp = self.client.post(endpoint, json=payload, timeout=self.timeout)
            if resp.status_code in (200, 201):
                return resp.json()
            else:
                logger.error(
                    f"Backend returned HTTP {resp.status_code} for {endpoint}: {resp.text}"
                )
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "error": resp.text,
                }
        except Exception as e:
            err_str = str(e).lower()
            if "connection" in err_str or "refused" in err_str or "failed" in err_str:
                logger.error(f"Backend unavailable at {self.base_url}. Ingestion skipped.")
                return {"success": False, "error": "Backend unavailable"}
            elif "timeout" in err_str:
                logger.error(f"Request to {endpoint} timed out after {self.timeout}s.")
                return {"success": False, "error": "Request timed out"}
            else:
                logger.error(f"Unexpected error posting to {endpoint}: {e}")
                return {"success": False, "error": str(e)}

