"""Pydantic schemas matching the actual NETRA Frontend contracts."""
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class CameraCreate(BaseModel):
    """Schema for camera registration."""
    id: str = Field(..., description="Camera ID (e.g. CAM-001, CAM-DL-412)")
    name: str = Field(..., description="Human-readable camera name")
    latitude: float = Field(..., description="WGS84 latitude")
    longitude: float = Field(..., description="WGS84 longitude")
    status: Optional[str] = Field("online", description="Status: online, warning, offline")
    location_name: Optional[str] = Field(None, description="Descriptive location")
    fps: Optional[int] = Field(60, description="Camera stream frame rate")
    speed_avg: Optional[str] = Field(None, description="Average speed string e.g. '38 km/h'")
    speed_limit_kmh: Optional[float] = Field(None, description="Corridor speed limit in km/h")
    road_name: Optional[str] = Field(None, description="Road or highway corridor name")
    last_detection: Optional[str] = Field(None, description="Last detection time string")
    vehicle_count: Optional[int] = Field(0, description="Current vehicle count")
    city: Optional[str] = Field("Coimbatore", description="City name")
    state: Optional[str] = Field("Tamil Nadu", description="State name")

    model_config = ConfigDict(from_attributes=True)


class CameraResponse(BaseModel):
    """Camera schema shaped exactly as consumed by frontend gisService.ts."""
    id: str
    name: str
    latitude: float
    longitude: float
    status: str
    lastDetection: Optional[str] = None
    vehicleCount: Optional[int] = 0
    locationName: Optional[str] = None
    speedAvg: Optional[str] = None
    speedLimitKmh: Optional[float] = None
    speed_limit_kmh: Optional[float] = None
    roadName: Optional[str] = None
    road_name: Optional[str] = None
    fps: Optional[int] = 60
    city: Optional[str] = "Coimbatore"
    state: Optional[str] = "Tamil Nadu"

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class VehicleDetectionResponse(BaseModel):
    """Detection item in vehicle trajectory consumed by TrackingPage.jsx."""
    cameraId: str
    cameraName: Optional[str] = None
    latitude: float
    longitude: float
    timestamp: str
    plate: str
    vehicleType: str
    confidence: float
    direction: Optional[str] = None
    speed: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class VehicleTrajectoryResponse(BaseModel):
    """Trajectory schema shaped exactly as consumed by TrackingPage.jsx."""
    plate: str
    status: Optional[str] = "Active Track"
    totalDistanceKm: Optional[float] = 0.0
    avgSpeedKmH: Optional[float] = 0.0
    detections: List[VehicleDetectionResponse] = []

    model_config = ConfigDict(from_attributes=True)


class TrafficZone(BaseModel):
    """Traffic zone GIS polygon/circle overlay."""
    id: str
    name: str
    status: str  # normal, congestion, heavy, incident
    statusText: str
    color: str
    center: Tuple[float, float]
    radiusMeters: int
    density: str
    avgSpeed: str

    model_config = ConfigDict(from_attributes=True)


class TrafficIncident(BaseModel):
    """Traffic incident alert pin."""
    id: str
    title: str
    severity: str  # low, medium, high, critical
    latitude: float
    longitude: float
    timestamp: str
    corridor: str
    description: str

    model_config = ConfigDict(from_attributes=True)


class TrafficEventsResponse(BaseModel):
    """Combined traffic zones and incidents response."""
    zones: List[TrafficZone]
    incidents: List[TrafficIncident]

    model_config = ConfigDict(from_attributes=True)


class PlateCaptureResponse(BaseModel):
    """ANPR capture item consumed by AnprPage.jsx."""
    id: int
    plate: str
    type: str
    camId: str
    timestamp: str
    confidence: str
    status: str

    model_config = ConfigDict(from_attributes=True)


class AlertResponse(BaseModel):
    """Enforcement alert item consumed by AlertsPage.jsx."""
    id: str
    type: str
    plate: str
    location: str
    severity: str
    time: str
    status: str

    model_config = ConfigDict(from_attributes=True)


class AlertCreate(BaseModel):
    """Schema for creating an automated enforcement alert."""
    id: Optional[str] = None
    alert_type: str
    severity: Optional[str] = "Medium"
    message: Optional[str] = None
    camera_id: Optional[str] = None
    global_vehicle_id: Optional[str] = None
    plate_text: Optional[str] = None
    location_text: Optional[str] = None
    timestamp_text: Optional[str] = None
    status: Optional[str] = "Pending Review"

    model_config = ConfigDict(from_attributes=True)


class TrafficIncidentCreate(BaseModel):
    """Schema for creating a traffic incident event."""
    id: Optional[str] = None
    event_type: Optional[str] = "incident"
    title: str
    severity: Optional[str] = "medium"
    camera_id: Optional[str] = None
    corridor: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: Optional[str] = None
    timestamp_text: Optional[str] = None
    status: Optional[str] = "active"

    model_config = ConfigDict(from_attributes=True)
