"""Vehicle records and trajectory API routes."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db.session import get_db
from app.models.entities import GlobalVehicle, VehicleObservation
from app.schemas.frontend import VehicleTrajectoryResponse
from app.services.trajectory_service import get_vehicle_trajectory_by_plate

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])


@router.get("/{plate}/trajectory", response_model=VehicleTrajectoryResponse)
def get_trajectory(plate: str, db: Session = Depends(get_db)):
    """Retrieve multi-camera trajectory for a specific vehicle license plate."""
    trajectory = get_vehicle_trajectory_by_plate(db, plate)
    if not trajectory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trajectory found for vehicle {plate}",
        )
    return trajectory


@router.get("", response_model=List[dict])
def list_global_vehicles(search: Optional[str] = None, skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """List canonical global vehicles with observation history and timeline."""
    query = select(GlobalVehicle)
    if search:
        s_clean = search.strip().replace(" ", "").replace("-", "").replace("_", "").upper()
        matched_gv_ids = set()
        all_obs = db.execute(select(VehicleObservation).where(VehicleObservation.plate_text.isnot(None))).scalars().all()
        for o in all_obs:
            norm_p = (o.plate_text or "").replace(" ", "").replace("-", "").replace("_", "").upper()
            if norm_p and (s_clean in norm_p or norm_p in s_clean):
                if o.global_vehicle_id:
                    matched_gv_ids.add(o.global_vehicle_id)

        query = query.where(
            (GlobalVehicle.global_vehicle_id.ilike(f"%{search.strip()}%")) |
            (GlobalVehicle.plate_text.ilike(f"%{search.strip()}%")) |
            (GlobalVehicle.global_vehicle_id.in_(matched_gv_ids))
        )

    vehicles = db.execute(
        query.order_by(GlobalVehicle.global_vehicle_id).offset(skip).limit(limit)
    ).scalars().all()
    res = []
    for v in vehicles:
        obs_sorted = sorted(v.observations, key=lambda o: (o.created_at or o.id)) if v.observations else []
        obs_cams = []
        for o in obs_sorted:
            if o.camera_id not in obs_cams:
                obs_cams.append(o.camera_id)

        obs_plate = next((o.plate_text for o in obs_sorted if o.plate_text), None)
        first_cam = obs_cams[0] if obs_cams else None
        last_cam = obs_cams[-1] if obs_cams else None

        timeline = []
        for idx, o in enumerate(obs_sorted):
            road = o.camera.road_name if o.camera and o.camera.road_name else (o.camera.location_name if o.camera else None)
            time_str = o.created_at.strftime("%H:%M:%S") if (o.created_at and hasattr(o.created_at, "strftime")) else "14:30:00"
            status_text = "First Detection" if idx == 0 else ("Last Seen" if idx == len(obs_sorted) - 1 else "Detected")
            timeline.append({
                "camera_id": o.camera_id,
                "camera_name": o.camera.name if o.camera else o.camera_id,
                "road_name": road or "Corridor Node",
                "timestamp": time_str,
                "status": status_text,
                "confidence": o.plate_confidence or o.best_ocr_confidence or 0.95
            })

        res.append({
            "global_vehicle_id": v.global_vehicle_id,
            "vehicle_class": v.vehicle_class,
            "plate_text": v.plate_text or obs_plate,
            "match_confidence": v.match_confidence,
            "camera_observations": obs_cams,
            "first_seen_cam": first_cam,
            "last_seen_cam": last_cam,
            "camera_count": len(obs_cams),
            "timeline": timeline,
            "created_at": v.created_at.isoformat() if (v.created_at and hasattr(v.created_at, "isoformat")) else None,
        })
    return res

