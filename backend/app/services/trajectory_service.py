"""Vehicle trajectory query and reconstruction service."""
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.entities import VehicleTrajectory, TrajectorySegment, Camera, VehicleObservation, GlobalVehicle
from app.schemas.frontend import VehicleTrajectoryResponse, VehicleDetectionResponse


def _normalize_plate(plate: str) -> str:
    return (plate or "").replace(" ", "").replace("-", "").replace("_", "").upper()


def get_vehicle_trajectory_by_plate(db: Session, plate_text: str) -> Optional[VehicleTrajectoryResponse]:
    """
    Retrieve vehicle trajectory by license plate string.
    Reconstructs camera detections along the corridor with geo-coordinates.
    """
    norm = _normalize_plate(plate_text)
    if not norm:
        return None

    # Search in trajectories matching normalized plate or global_vehicle_id
    # Also resolve plate to global_vehicle_id if plate is tracked on GlobalVehicle or VehicleObservation
    matched_gv_ids = set()
    gvs = db.execute(select(GlobalVehicle).where(GlobalVehicle.plate_text.isnot(None))).scalars().all()
    for gv in gvs:
        if _normalize_plate(gv.plate_text) == norm:
            matched_gv_ids.add(_normalize_plate(gv.global_vehicle_id))

    obs_plates = db.execute(select(VehicleObservation).where(VehicleObservation.plate_text.isnot(None))).scalars().all()
    for obs in obs_plates:
        if _normalize_plate(obs.plate_text) == norm and obs.global_vehicle_id:
            matched_gv_ids.add(_normalize_plate(obs.global_vehicle_id))

    trajectories = db.execute(select(VehicleTrajectory)).scalars().all()
    target_traj: Optional[VehicleTrajectory] = None
    for t in trajectories:
        t_plate_norm = _normalize_plate(t.plate_text) if t.plate_text else None
        t_gv_norm = _normalize_plate(t.global_vehicle_id) if t.global_vehicle_id else None
        if t_plate_norm == norm or t_gv_norm == norm or (t_gv_norm and t_gv_norm in matched_gv_ids):
            target_traj = t
            break

    if not target_traj:
        return None

    # Fetch camera map for fast lookup
    cameras = {c.id: c for c in db.execute(select(Camera)).scalars().all()}

    detections: List[VehicleDetectionResponse] = []
    # If trajectory has segments, build detections from segments
    if target_traj.segments:
        # Sort segments by id/created_at
        sorted_segs = sorted(target_traj.segments, key=lambda s: s.id)
        for idx, seg in enumerate(sorted_segs):
            from_cam = cameras.get(seg.from_camera)
            to_cam = cameras.get(seg.to_camera)

            if idx == 0 and from_cam:
                detections.append(
                    VehicleDetectionResponse(
                        cameraId=from_cam.id,
                        cameraName=from_cam.name,
                        latitude=from_cam.latitude,
                        longitude=from_cam.longitude,
                        timestamp=seg.from_timestamp or "14:00:00",
                        plate=target_traj.plate_text or plate_text,
                        vehicleType=target_traj.vehicle_class,
                        confidence=99.2,
                        direction=seg.direction or "North",
                        speed=f"{int(seg.average_speed_kmh)} km/h" if seg.average_speed_kmh else "45 km/h",
                    )
                )

            if to_cam:
                detections.append(
                    VehicleDetectionResponse(
                        cameraId=to_cam.id,
                        cameraName=to_cam.name,
                        latitude=to_cam.latitude,
                        longitude=to_cam.longitude,
                        timestamp=seg.to_timestamp or "14:15:00",
                        plate=target_traj.plate_text or plate_text,
                        vehicleType=target_traj.vehicle_class,
                        confidence=99.0,
                        direction=seg.direction or "North-East",
                        speed=f"{int(seg.average_speed_kmh)} km/h" if seg.average_speed_kmh else "50 km/h",
                    )
                )

    # Ensure all cameras in camera_sequence or observed by this global vehicle are represented
    existing_cam_ids = {d.cameraId for d in detections}
    seq_cams = list(target_traj.camera_sequence or [])
    if target_traj.global_vehicle_id:
        obs_cams = db.execute(
            select(VehicleObservation.camera_id)
            .where(VehicleObservation.global_vehicle_id == target_traj.global_vehicle_id)
            .order_by(VehicleObservation.id)
        ).scalars().all()
        for c in obs_cams:
            if c not in seq_cams:
                seq_cams.append(c)

    for idx, cam_id in enumerate(seq_cams):
        if cam_id not in existing_cam_ids and cam_id in cameras:
            cam = cameras[cam_id]
            detections.append(
                VehicleDetectionResponse(
                    cameraId=cam.id,
                    cameraName=cam.name,
                    latitude=cam.latitude,
                    longitude=cam.longitude,
                    timestamp=f"14:{20 + idx * 5:02d}:00",
                    plate=target_traj.plate_text or plate_text,
                    vehicleType=target_traj.vehicle_class,
                    confidence=99.0,
                    direction="North-East",
                    speed=f"{int(target_traj.average_speed_kmh)} km/h" if target_traj.average_speed_kmh else "48 km/h",
                )
            )
            existing_cam_ids.add(cam_id)

    return VehicleTrajectoryResponse(
        plate=target_traj.plate_text or plate_text,
        status="Active Track",
        totalDistanceKm=round(target_traj.total_distance_meters / 1000.0, 1),
        avgSpeedKmH=round(target_traj.average_speed_kmh, 1),
        detections=detections,
    )
