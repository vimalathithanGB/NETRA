"""SQLAlchemy Models for NETRA Traffic Intelligence System.

Covers all 9 confirmed domain entities:
1. Camera (surveillance camera nodes and geo-locations with PostGIS)
2. GlobalVehicle (canonical vehicle entity across multi-camera grid)
3. VehicleObservation (per-camera detection tracks from AI engine)
4. LicensePlate (ANPR plate OCR index and status tracking)
5. CrossCameraMatch (Re-ID and plate match correlations between camera pairs)
6. VehicleTrajectory (multi-camera spatial corridors)
7. TrajectorySegment (individual camera-to-camera movement segments)
8. RoutePrediction (transition probability to next cameras)
9. CameraTransition (historical transition metrics between camera nodes)
"""
from typing import List, Optional, Any
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    UniqueConstraint,
    Index,
    func
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from geoalchemy2 import Geometry
from app.db.base import Base


class Camera(Base):
    """Surveillance camera node with geo-spatial coordinates."""
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    location_geom = Column(Geometry(geometry_type="POINT", srid=4326, spatial_index=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="online", nullable=False)
    location_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    fps: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    vehicle_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    speed_avg: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    speed_limit_kmh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    road_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    last_detection: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    observations: Mapped[List["VehicleObservation"]] = relationship("VehicleObservation", back_populates="camera")


class GlobalVehicle(Base):
    """Canonical vehicle entity matched across cameras."""
    __tablename__ = "global_vehicles"

    global_vehicle_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    vehicle_class: Mapped[str] = mapped_column(String(64), nullable=False)
    plate_text: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    match_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    observations: Mapped[List["VehicleObservation"]] = relationship("VehicleObservation", back_populates="global_vehicle", passive_deletes=True)
    trajectories: Mapped[List["VehicleTrajectory"]] = relationship("VehicleTrajectory", back_populates="global_vehicle", cascade="all, delete-orphan", passive_deletes=True)
    predictions: Mapped[List["RoutePrediction"]] = relationship("RoutePrediction", back_populates="global_vehicle", cascade="all, delete-orphan", passive_deletes=True)


class VehicleObservation(Base):
    """Track observation event at a specific camera."""
    __tablename__ = "vehicle_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    global_vehicle_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("global_vehicles.global_vehicle_id", ondelete="SET NULL"), nullable=True, index=True)
    vehicle_class: Mapped[str] = mapped_column(String(64), nullable=False)
    plate_text: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    plate_status: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    plate_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    observation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    valid_observation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    best_ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_bbox: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    detector_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    first_seen_frame: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_seen_frame: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="observations")
    global_vehicle: Mapped[Optional["GlobalVehicle"]] = relationship("GlobalVehicle", back_populates="observations")

    __table_args__ = (
        Index("ix_obs_camera_track", "camera_id", "track_id"),
    )


class LicensePlate(Base):
    """Plate OCR registry and historical sightings index."""
    __tablename__ = "license_plates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_text: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    plate_status: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vehicle_class: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_camera_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    last_seen_timestamp: Mapped[Optional[Any]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class CrossCameraMatch(Base):
    """Cross-camera Re-ID match evidence and score."""
    __tablename__ = "cross_camera_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_a: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    track_a: Mapped[int] = mapped_column(Integer, nullable=False)
    camera_b: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    track_b: Mapped[int] = mapped_column(Integer, nullable=False)
    global_vehicle_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("global_vehicles.global_vehicle_id", ondelete="SET NULL"), nullable=True, index=True)
    reid_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    plate_match: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    class_match: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    available_evidence: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    match_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    matched: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_match_cameras", "camera_a", "camera_b"),
    )


class VehicleTrajectory(Base):
    """Spatial-temporal vehicle trajectory across multiple camera nodes."""
    __tablename__ = "vehicle_trajectories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_vehicle_id: Mapped[str] = mapped_column(String(64), ForeignKey("global_vehicles.global_vehicle_id", ondelete="CASCADE"), nullable=False, index=True)
    vehicle_class: Mapped[str] = mapped_column(String(64), nullable=False)
    plate_text: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    camera_sequence: Mapped[Any] = mapped_column(JSON, nullable=False)
    total_distance_meters: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_travel_time_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    average_speed_kmh: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    global_vehicle: Mapped["GlobalVehicle"] = relationship("GlobalVehicle", back_populates="trajectories")
    segments: Mapped[List["TrajectorySegment"]] = relationship("TrajectorySegment", back_populates="trajectory", cascade="all, delete-orphan")


class TrajectorySegment(Base):
    """Segment between two consecutive cameras in a vehicle trajectory."""
    __tablename__ = "trajectory_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trajectory_id: Mapped[int] = mapped_column(Integer, ForeignKey("vehicle_trajectories.id", ondelete="CASCADE"), nullable=False, index=True)
    from_camera: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    to_camera: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    from_timestamp: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    to_timestamp: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    distance_meters: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cumulative_distance_meters: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    travel_time_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    average_speed_kmh: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    speed_limit_kmh: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    minimum_feasible_time_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    travel_time_feasible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    bearing_degrees: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    direction: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    match_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reid_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    available_evidence: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    trajectory: Mapped["VehicleTrajectory"] = relationship("VehicleTrajectory", back_populates="segments")


class RoutePrediction(Base):
    """Predicted next camera destination for an active global vehicle."""
    __tablename__ = "route_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_vehicle_id: Mapped[str] = mapped_column(String(64), ForeignKey("global_vehicles.global_vehicle_id", ondelete="CASCADE"), nullable=False, index=True)
    current_camera: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    vehicle_class: Mapped[str] = mapped_column(String(64), nullable=False)
    prediction_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    predictions: Mapped[Any] = mapped_column(JSON, nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    global_vehicle: Mapped["GlobalVehicle"] = relationship("GlobalVehicle", back_populates="predictions")


class CameraTransition(Base):
    """Statistical transitions between connected camera nodes."""
    __tablename__ = "camera_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_camera: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    to_camera: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    transition_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    avg_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    distance_meters: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("from_camera", "to_camera", name="uq_from_to_camera"),
    )


class Alert(Base):
    """Traffic violation and critical enforcement alert."""
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), default="Medium", nullable=False)
    message: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    camera_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    global_vehicle_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("global_vehicles.global_vehicle_id", ondelete="SET NULL"), nullable=True)
    plate_text: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    location_text: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    timestamp_text: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="Pending Review", nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    camera: Mapped[Optional["Camera"]] = relationship("Camera")
    global_vehicle: Mapped[Optional["GlobalVehicle"]] = relationship("GlobalVehicle")


class TrafficEvent(Base):
    """Traffic zone, density corridor, or incident pin."""
    __tablename__ = "traffic_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), default="medium", nullable=False)
    camera_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    corridor: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    location_geom = Column(Geometry(geometry_type="POINT", srid=4326, spatial_index=True), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    timestamp_text: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    camera: Mapped[Optional["Camera"]] = relationship("Camera")

