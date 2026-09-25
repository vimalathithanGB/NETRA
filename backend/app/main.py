"""NETRA Backend FastAPI Application Entrypoint."""
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router
from app.api.v1.cameras import router as cameras_router
from app.api.v1.vehicles import router as vehicles_router
from app.api.v1.traffic import router as traffic_router
from app.api.v1.plates import router as plates_router
from app.api.v1.alerts import router as alerts_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.video_testing import router as video_testing_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Networked Engine for Traffic Recognition & Analytics - Government of India MoRTH Backend Service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Configuration
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# System Health Check Endpoint
@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint confirming system operational status."""
    return {"status": "ok"}


# Canonical API v1 Router Mount
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# ============================================================================
# FRONTEND GIS COMPATIBILITY BRIDGE
# The frontend gisService.ts calls /api/cameras, /api/vehicles/{plate}/trajectory,
# /api/traffic/events, /api/plates/captures, /api/alerts, /api/analytics without the /v1 version segment.
# To ensure seamless zero-configuration operation with the existing frontend
# while keeping /api/v1 as the canonical API version, these routes are mounted
# under /api as documented compatibility aliases.
# ============================================================================
compat_router = APIRouter(prefix="/api", include_in_schema=False)
compat_router.include_router(cameras_router)
compat_router.include_router(vehicles_router)
compat_router.include_router(traffic_router)
compat_router.include_router(plates_router)
compat_router.include_router(alerts_router)
compat_router.include_router(analytics_router)
compat_router.include_router(video_testing_router)
app.include_router(compat_router)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
