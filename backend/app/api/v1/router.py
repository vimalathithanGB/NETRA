"""API v1 Master Router.

Registers all NETRA v1 endpoints:
- /api/v1/status     -> Operational status check
- /api/v1/cameras    -> Surveillance camera metadata and telemetry
- /api/v1/vehicles   -> Global vehicle records and trajectory
- /api/v1/ingest     -> AI Engine output ingestion pipeline
- /api/v1/plates     -> ANPR plate captures and OCR lookup
- /api/v1/traffic    -> Traffic density zones and incident telemetry
- /api/v1/alerts     -> Critical violations and e-Challan enforcement
"""
from fastapi import APIRouter
from app.api.v1.cameras import router as cameras_router
from app.api.v1.detections import router as detections_router
from app.api.v1.vehicles import router as vehicles_router
from app.api.v1.traffic import router as traffic_router
from app.api.v1.plates import router as plates_router
from app.api.v1.alerts import router as alerts_router
from app.api.v1.analytics import router as analytics_router

api_router = APIRouter()


@api_router.get("/status", tags=["System"])
def get_api_v1_status():
    """Returns the operational status of API v1."""
    return {
        "api_version": "v1",
        "status": "ready",
        "modules": {
            "auth": "prepared",
            "cameras": "ready",
            "vehicles": "ready",
            "detections": "waiting_for_ai_json_contract",
            "plates": "ready",
            "trajectories": "ready",
            "analytics": "ready",
            "alerts": "ready",
        }

    }


# Mount feature routers
api_router.include_router(cameras_router)
api_router.include_router(detections_router)
api_router.include_router(vehicles_router)
api_router.include_router(traffic_router)
api_router.include_router(plates_router)
api_router.include_router(alerts_router)
api_router.include_router(analytics_router)

