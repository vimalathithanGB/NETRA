"""Camera management and GIS telemetry service."""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.entities import Camera
from app.schemas.frontend import CameraCreate, CameraResponse

from pathlib import Path
import yaml

def _load_cameras_config() -> List[dict]:
    """Load camera definitions from configs/cameras.yaml with accurate coordinates."""
    possible_paths = [
        Path(__file__).resolve().parent.parent.parent.parent / "configs" / "cameras.yaml",
        Path("configs/cameras.yaml"),
        Path("../configs/cameras.yaml"),
    ]
    for p in possible_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data and "cameras" in data:
                        cams = []
                        for cid, cinfo in data["cameras"].items():
                            cams.append({
                                "id": cinfo.get("camera_id", cid),
                                "name": f"{cid} - {cinfo.get('road_name', 'Sensor Node')}",
                                "latitude": float(cinfo["latitude"]),
                                "longitude": float(cinfo["longitude"]),
                                "status": "online",
                                "last_detection": "14:30:00",
                                "vehicle_count": 0,
                                "location_name": cinfo.get("description", cinfo.get("road_name")),
                                "road_name": cinfo.get("road_name"),
                                "speed_limit_kmh": float(cinfo.get("speed_limit_kmh", 40.0)),
                                "speed_avg": f"{cinfo.get('speed_limit_kmh', 40)} km/h",
                                "fps": 30,
                            })
                        return cams
            except Exception:
                pass

    return [
        {
            "id": "CAM_01",
            "name": "CAM_01 - SIMULATED_ROAD_A",
            "latitude": 11.0168,
            "longitude": 76.9558,
            "status": "online",
            "last_detection": "14:30:00",
            "vehicle_count": 31,
            "location_name": "Ingress Corridor Sensor Node (Simulated Junction North)",
            "road_name": "SIMULATED_ROAD_A",
            "speed_limit_kmh": 40.0,
            "speed_avg": "40 km/h",
            "fps": 30,
        },
        {
            "id": "CAM_02",
            "name": "CAM_02 - SIMULATED_ROAD_B",
            "latitude": 11.0192,
            "longitude": 76.9581,
            "status": "online",
            "last_detection": "14:30:00",
            "vehicle_count": 24,
            "location_name": "Mid-Corridor Surveillance Node (Simulated Central Avenue)",
            "road_name": "SIMULATED_ROAD_B",
            "speed_limit_kmh": 40.0,
            "speed_avg": "40 km/h",
            "fps": 30,
        },
        {
            "id": "CAM_03",
            "name": "CAM_03 - SIMULATED_ROAD_C",
            "latitude": 11.0225,
            "longitude": 76.9610,
            "status": "online",
            "last_detection": "14:30:00",
            "vehicle_count": 17,
            "location_name": "Egress Sensor Node (Simulated Junction South)",
            "road_name": "SIMULATED_ROAD_C",
            "speed_limit_kmh": 40.0,
            "speed_avg": "40 km/h",
            "fps": 30,
        },
    ]

DEFAULT_CAMERAS = _load_cameras_config()



def _format_camera_response(cam: Camera) -> CameraResponse:
    """Format SQLAlchemy Camera to Frontend response shape."""
    return CameraResponse(
        id=cam.id,
        name=cam.name,
        latitude=cam.latitude,
        longitude=cam.longitude,
        status=cam.status,
        lastDetection=cam.last_detection,
        vehicleCount=cam.vehicle_count,
        locationName=cam.location_name,
        speedAvg=cam.speed_avg,
        roadName=cam.road_name,
        road_name=cam.road_name,
        speedLimitKmh=cam.speed_limit_kmh,
        speed_limit_kmh=cam.speed_limit_kmh,
        fps=cam.fps,
        city="Coimbatore",
        state="Tamil Nadu",
    )



def get_all_cameras(db: Session) -> List[CameraResponse]:
    """Return all surveillance cameras. If empty, auto-seed standard nodes."""
    cameras = db.execute(select(Camera).order_by(Camera.id)).scalars().all()
    if not cameras:
        seed_default_cameras(db)
        cameras = db.execute(select(Camera).order_by(Camera.id)).scalars().all()
    return [_format_camera_response(c) for c in cameras]


def get_camera(db: Session, camera_id: str) -> Optional[Camera]:
    """Retrieve camera by primary key."""
    return db.get(Camera, camera_id)


def create_or_update_camera(db: Session, camera_in: CameraCreate) -> CameraResponse:
    """Insert or update camera record with PostGIS point geometry."""
    cam = db.get(Camera, camera_in.id)
    wkt_geom = f"SRID=4326;POINT({camera_in.longitude} {camera_in.latitude})"
    loc_name = camera_in.location_name
    if not loc_name:
        road_part = camera_in.road_name or camera_in.name
        city_part = camera_in.city or "Coimbatore"
        state_part = camera_in.state or "Tamil Nadu"
        loc_name = f"{road_part}, {city_part}, {state_part}"

    if cam:
        cam.name = camera_in.name
        cam.latitude = camera_in.latitude
        cam.longitude = camera_in.longitude
        cam.location_geom = wkt_geom
        cam.status = camera_in.status
        cam.location_name = loc_name
        cam.road_name = camera_in.road_name
        cam.speed_limit_kmh = camera_in.speed_limit_kmh
        cam.fps = camera_in.fps
        cam.speed_avg = camera_in.speed_avg
        cam.last_detection = camera_in.last_detection
        cam.vehicle_count = camera_in.vehicle_count
    else:
        cam = Camera(
            id=camera_in.id,
            name=camera_in.name,
            latitude=camera_in.latitude,
            longitude=camera_in.longitude,
            location_geom=wkt_geom,
            status=camera_in.status,
            location_name=loc_name,
            road_name=camera_in.road_name,
            speed_limit_kmh=camera_in.speed_limit_kmh,
            fps=camera_in.fps,
            speed_avg=camera_in.speed_avg,
            last_detection=camera_in.last_detection,
            vehicle_count=camera_in.vehicle_count,
        )
        db.add(cam)
    db.commit()
    db.refresh(cam)
    return _format_camera_response(cam)



def seed_default_cameras(db: Session) -> int:
    """Seed initial surveillance nodes into the database if not present, and update existing with road metadata."""
    count = 0
    for node in DEFAULT_CAMERAS:
        cam = db.get(Camera, node["id"])
        wkt_geom = f"SRID=4326;POINT({node['longitude']} {node['latitude']})"
        if not cam:
            cam = Camera(
                id=node["id"],
                name=node["name"],
                latitude=node["latitude"],
                longitude=node["longitude"],
                location_geom=wkt_geom,
                status=node["status"],
                location_name=node["location_name"],
                road_name=node.get("road_name"),
                speed_limit_kmh=node.get("speed_limit_kmh"),
                fps=node["fps"],
                speed_avg=node["speed_avg"],
                last_detection=node["last_detection"],
                vehicle_count=node["vehicle_count"],
            )
            db.add(cam)
            count += 1
        else:
            # Sync road_name and speed_limit_kmh if missing
            if not cam.road_name and node.get("road_name"):
                cam.road_name = node.get("road_name")
                count += 1
            if not cam.speed_limit_kmh and node.get("speed_limit_kmh"):
                cam.speed_limit_kmh = node.get("speed_limit_kmh")
                count += 1
    if count > 0:
        db.commit()
    return count
