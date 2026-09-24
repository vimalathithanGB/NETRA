"""Service Layer Registry for NETRA."""
from app.services.camera_service import (
    get_all_cameras,
    get_camera,
    create_or_update_camera,
    seed_default_cameras,
)
from app.services.ingest_service import (
    ingest_tracks,
    ingest_matches,
    ingest_global_vehicles,
    ingest_trajectories,
    ingest_predictions,
    ingest_unified,
)
from app.services.trajectory_service import (
    get_vehicle_trajectory_by_plate,
)
from app.services.traffic_service import (
    get_traffic_events,
    get_recent_plate_captures,
    get_active_alerts,
)

__all__ = [
    "get_all_cameras",
    "get_camera",
    "create_or_update_camera",
    "seed_default_cameras",
    "ingest_tracks",
    "ingest_matches",
    "ingest_global_vehicles",
    "ingest_trajectories",
    "ingest_predictions",
    "ingest_unified",
    "get_vehicle_trajectory_by_plate",
    "get_traffic_events",
    "get_recent_plate_captures",
    "get_active_alerts",
]
