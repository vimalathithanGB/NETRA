"""SQLAlchemy Model Registry.

Registers and exports all 9 core domain entities for NETRA.
"""
from app.db.base import Base
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
    Alert,
    TrafficEvent,
)

__all__ = [
    "Base",
    "Camera",
    "GlobalVehicle",
    "VehicleObservation",
    "LicensePlate",
    "CrossCameraMatch",
    "VehicleTrajectory",
    "TrajectorySegment",
    "RoutePrediction",
    "CameraTransition",
    "Alert",
    "TrafficEvent",
]
