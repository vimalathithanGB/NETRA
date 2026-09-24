"""AI Engine Ingestion Service.

Processes and stores confirmed AI Engine pipeline outputs:
- Camera Vehicle Tracks (VehicleObservation)
- Cross-Camera Matches (CrossCameraMatch)
- Global Vehicles (GlobalVehicle)
- Trajectories with segments (VehicleTrajectory, TrajectorySegment)
- Route Predictions (RoutePrediction)
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import select
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
)
from app.schemas.ai_contracts import (
    VehicleTrackInput,
    CrossCameraMatchInput,
    GlobalVehicleInput,
    TrajectoryInput,
    RoutePredictionInput,
    UnifiedAIIngestPayload,
)


def _ensure_camera_exists(db: Session, camera_id: str) -> None:
    """Ensure foreign key target camera exists, creating from configuration or stub if new."""
    cam = db.get(Camera, camera_id)
    if not cam:
        from app.services.camera_service import DEFAULT_CAMERAS
        cam_meta = next((c for c in DEFAULT_CAMERAS if c["id"] == camera_id), None)
        if cam_meta:
            lat = cam_meta["latitude"]
            lon = cam_meta["longitude"]
            name = cam_meta["name"]
            loc_name = cam_meta.get("location_name", name)
        else:
            lat = 11.0168
            lon = 76.9558
            name = f"Camera {camera_id}"
            loc_name = f"Sensor Node {camera_id}"

        stub = Camera(
            id=camera_id,
            name=name,
            latitude=lat,
            longitude=lon,
            location_geom=f"SRID=4326;POINT({lon} {lat})",
            location_name=loc_name,
            status="online",
            fps=30,
        )
        db.add(stub)
        db.flush()



def _sync_license_plate(
    db: Session,
    plate_text: Optional[str],
    plate_status: Optional[str] = "unknown",
    confidence: Optional[float] = None,
    vehicle_class: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> None:
    """Synchronize OCR plate sighting in the license_plates index table."""
    if not plate_text:
        return
    norm_plate = plate_text.replace(" ", "").replace("-", "").upper()
    if not norm_plate:
        return

    plate_record = db.execute(
        select(LicensePlate).where(LicensePlate.plate_text == norm_plate)
    ).scalars().first()

    now = datetime.now(timezone.utc)
    if plate_record:
        if confidence and (not plate_record.confidence or confidence > plate_record.confidence):
            plate_record.confidence = confidence
        if plate_status and plate_status != "unknown":
            plate_record.plate_status = plate_status
        if vehicle_class:
            plate_record.vehicle_class = vehicle_class
        if camera_id:
            plate_record.last_camera_id = camera_id
        plate_record.last_seen_timestamp = now
    else:
        plate_record = LicensePlate(
            plate_text=norm_plate,
            plate_status=plate_status or "unknown",
            confidence=confidence,
            vehicle_class=vehicle_class,
            last_camera_id=camera_id,
            last_seen_timestamp=now,
        )
        db.add(plate_record)
    db.flush()


def ingest_tracks(db: Session, camera_id: str, tracks: List[VehicleTrackInput], commit: bool = True) -> int:
    """Ingest a batch of vehicle tracks observed at a camera node."""
    _ensure_camera_exists(db, camera_id)
    stored_count = 0

    for t in tracks:
        # Check if observation for camera and track_id already exists
        obs = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == camera_id,
                VehicleObservation.track_id == t.track_id,
            )
        ).scalars().first()

        if obs:
            obs.vehicle_class = t.vehicle_class
            obs.plate_text = t.plate_text
            obs.plate_status = t.plate_status or "unknown"
            obs.plate_confidence = t.plate_confidence
            obs.observation_count = t.observation_count or 0
            obs.valid_observation_count = t.valid_observation_count or 0
            obs.best_ocr_confidence = t.best_ocr_confidence
            obs.last_bbox = t.last_bbox
            obs.detector_confidence = t.detector_confidence
            obs.first_seen_frame = t.first_seen_frame
            obs.last_seen_frame = t.last_seen_frame
        else:
            obs = VehicleObservation(
                camera_id=camera_id,
                track_id=t.track_id,
                vehicle_class=t.vehicle_class,
                plate_text=t.plate_text,
                plate_status=t.plate_status or "unknown",
                plate_confidence=t.plate_confidence,
                observation_count=t.observation_count or 0,
                valid_observation_count=t.valid_observation_count or 0,
                best_ocr_confidence=t.best_ocr_confidence,
                last_bbox=t.last_bbox,
                detector_confidence=t.detector_confidence,
                first_seen_frame=t.first_seen_frame,
                last_seen_frame=t.last_seen_frame,
            )
            db.add(obs)

        stored_count += 1

        if t.plate_text:
            _sync_license_plate(
                db,
                plate_text=t.plate_text,
                plate_status=t.plate_status,
                confidence=t.best_ocr_confidence or t.plate_confidence,
                vehicle_class=t.vehicle_class,
                camera_id=camera_id,
            )

    if commit:
        db.commit()
    else:
        db.flush()
    return stored_count


def ingest_matches(db: Session, matches: List[CrossCameraMatchInput], commit: bool = True) -> int:
    """Ingest cross-camera Re-ID match pairs."""
    stored_count = 0
    transitions_map: Dict[Tuple[str, str], int] = {}

    for m in matches:
        _ensure_camera_exists(db, m.camera_a)
        _ensure_camera_exists(db, m.camera_b)

        existing_list = db.execute(
            select(CrossCameraMatch).where(
                CrossCameraMatch.camera_a == m.camera_a,
                CrossCameraMatch.track_a == m.track_a,
                CrossCameraMatch.camera_b == m.camera_b,
                CrossCameraMatch.track_b == m.track_b,
            )
        ).scalars().all()

        if existing_list:
            existing = existing_list[0]
            existing.reid_similarity = m.reid_similarity if m.reid_similarity is not None else 0.0
            existing.plate_match = bool(m.plate_match) if m.plate_match is not None else False
            existing.class_match = bool(m.class_match) if m.class_match is not None else False
            existing.available_evidence = m.available_evidence
            existing.match_score = m.match_score
            existing.threshold = m.threshold
            existing.matched = m.matched
            for dup in existing_list[1:]:
                db.delete(dup)
        else:
            record = CrossCameraMatch(
                camera_a=m.camera_a,
                track_a=m.track_a,
                camera_b=m.camera_b,
                track_b=m.track_b,
                reid_similarity=m.reid_similarity if m.reid_similarity is not None else 0.0,
                plate_match=bool(m.plate_match) if m.plate_match is not None else False,
                class_match=bool(m.class_match) if m.class_match is not None else False,
                available_evidence=m.available_evidence,
                match_score=m.match_score,
                threshold=m.threshold,
                matched=m.matched,
            )
            db.add(record)

        stored_count += 1

        # Track transitions for matched pairs across different cameras
        if m.matched and m.camera_a != m.camera_b:
            pair = (m.camera_a, m.camera_b)
            transitions_map[pair] = transitions_map.get(pair, 0) + 1

    # Batch update camera transitions without duplicate key violations
    for (from_cam, to_cam), count in transitions_map.items():
        trans = db.execute(
            select(CameraTransition).where(
                CameraTransition.from_camera == from_cam,
                CameraTransition.to_camera == to_cam,
            )
        ).scalars().first()
        if trans:
            trans.transition_count = max(trans.transition_count, count)
        else:
            trans = CameraTransition(
                from_camera=from_cam,
                to_camera=to_cam,
                transition_count=count,
                avg_duration_seconds=0.0,
                distance_meters=0.0,
            )
            db.add(trans)

    if commit:
        db.commit()
    else:
        db.flush()
    return stored_count


def ingest_global_vehicles(db: Session, vehicles: List[GlobalVehicleInput], commit: bool = True) -> int:
    """Ingest canonical global vehicles and link observations."""
    stored_count = 0
    for gv_in in vehicles:
        gv = db.get(GlobalVehicle, gv_in.global_vehicle_id)
        if gv:
            gv.vehicle_class = gv_in.vehicle_class
            gv.plate_text = gv_in.plate_text
            gv.match_confidence = gv_in.match_confidence
        else:
            gv = GlobalVehicle(
                global_vehicle_id=gv_in.global_vehicle_id,
                vehicle_class=gv_in.vehicle_class,
                plate_text=gv_in.plate_text,
                match_confidence=gv_in.match_confidence,
            )
            db.add(gv)
        stored_count += 1
        db.flush()

        # Link observations
        if gv_in.observations:
            for obs_ref in gv_in.observations:
                _ensure_camera_exists(db, obs_ref.camera_id)
                obs = db.execute(
                    select(VehicleObservation).where(
                        VehicleObservation.camera_id == obs_ref.camera_id,
                        VehicleObservation.track_id == obs_ref.track_id,
                    )
                ).scalars().first()
                if obs:
                    obs.global_vehicle_id = gv.global_vehicle_id
                else:
                    # Create placeholder observation linked to global vehicle
                    new_obs = VehicleObservation(
                        camera_id=obs_ref.camera_id,
                        track_id=obs_ref.track_id,
                        global_vehicle_id=gv.global_vehicle_id,
                        vehicle_class=gv_in.vehicle_class,
                        plate_text=gv_in.plate_text,
                    )
                    db.add(new_obs)

        if gv_in.plate_text:
            _sync_license_plate(
                db,
                plate_text=gv_in.plate_text,
                confidence=gv_in.match_confidence,
                vehicle_class=gv_in.vehicle_class,
            )

    if commit:
        db.commit()
    else:
        db.flush()
    return stored_count


def ingest_trajectories(db: Session, trajectories: List[TrajectoryInput], commit: bool = True) -> int:
    """Ingest spatial trajectories with constituent camera segments."""
    stored_count = 0
    for traj_in in trajectories:
        # Ensure parent GlobalVehicle exists
        gv = db.get(GlobalVehicle, traj_in.global_vehicle_id)
        if not gv:
            gv = GlobalVehicle(
                global_vehicle_id=traj_in.global_vehicle_id,
                vehicle_class=traj_in.vehicle_class,
                plate_text=traj_in.plate_text,
            )
            db.add(gv)
            db.flush()

        # Remove previous trajectories for this vehicle to update with newest
        existing_trajs = db.execute(
            select(VehicleTrajectory).where(
                VehicleTrajectory.global_vehicle_id == traj_in.global_vehicle_id
            )
        ).scalars().all()
        for et in existing_trajs:
            db.delete(et)
        if existing_trajs:
            db.flush()

        traj = VehicleTrajectory(
            global_vehicle_id=traj_in.global_vehicle_id,
            vehicle_class=traj_in.vehicle_class,
            plate_text=traj_in.plate_text or (gv.plate_text if gv else None),
            camera_sequence=traj_in.camera_sequence,
            total_distance_meters=traj_in.total_distance_meters,
            total_travel_time_seconds=traj_in.total_travel_time_seconds,
            average_speed_kmh=traj_in.average_speed_kmh,
        )
        db.add(traj)
        db.flush()

        if traj_in.segments:
            for seg_in in traj_in.segments:
                _ensure_camera_exists(db, seg_in.from_camera)
                _ensure_camera_exists(db, seg_in.to_camera)
                seg = TrajectorySegment(
                    trajectory_id=traj.id,
                    from_camera=seg_in.from_camera,
                    to_camera=seg_in.to_camera,
                    from_timestamp=seg_in.from_timestamp,
                    to_timestamp=seg_in.to_timestamp,
                    distance_meters=seg_in.distance_meters,
                    cumulative_distance_meters=seg_in.cumulative_distance_meters,
                    travel_time_seconds=seg_in.travel_time_seconds,
                    average_speed_kmh=seg_in.average_speed_kmh,
                    speed_limit_kmh=seg_in.speed_limit_kmh,
                    minimum_feasible_time_seconds=seg_in.minimum_feasible_time_seconds,
                    travel_time_feasible=seg_in.travel_time_feasible,
                    bearing_degrees=seg_in.bearing_degrees,
                    direction=seg_in.direction,
                    match_score=seg_in.match_score,
                    reid_similarity=seg_in.reid_similarity,
                    available_evidence=seg_in.available_evidence,
                )
                db.add(seg)

        if traj_in.plate_text:
            _sync_license_plate(
                db,
                plate_text=traj_in.plate_text,
                vehicle_class=traj_in.vehicle_class,
            )

        stored_count += 1

    if commit:
        db.commit()
    else:
        db.flush()
    return stored_count


def ingest_predictions(db: Session, predictions: List[RoutePredictionInput], commit: bool = True) -> int:
    """Ingest next-camera destination route predictions."""
    stored_count = 0
    for pred_in in predictions:
        gv = db.get(GlobalVehicle, pred_in.global_vehicle_id)
        if not gv:
            gv = GlobalVehicle(
                global_vehicle_id=pred_in.global_vehicle_id,
                vehicle_class=pred_in.vehicle_class,
            )
            db.add(gv)
            db.flush()

        _ensure_camera_exists(db, pred_in.current_camera)

        existing_list = db.execute(
            select(RoutePrediction).where(
                RoutePrediction.global_vehicle_id == pred_in.global_vehicle_id,
                RoutePrediction.current_camera == pred_in.current_camera,
            )
        ).scalars().all()

        if existing_list:
            existing = existing_list[0]
            existing.vehicle_class = pred_in.vehicle_class
            existing.prediction_available = pred_in.prediction_available
            existing.predictions = [p.model_dump() for p in pred_in.predictions]
            for dup in existing_list[1:]:
                db.delete(dup)
        else:
            record = RoutePrediction(
                global_vehicle_id=pred_in.global_vehicle_id,
                current_camera=pred_in.current_camera,
                vehicle_class=pred_in.vehicle_class,
                prediction_available=pred_in.prediction_available,
                predictions=[p.model_dump() for p in pred_in.predictions],
            )
            db.add(record)
        stored_count += 1

    if commit:
        db.commit()
    else:
        db.flush()
    return stored_count


def ingest_unified(db: Session, payload: UnifiedAIIngestPayload) -> Dict[str, Any]:
    """Process all AI outputs in a single transaction-safe call."""
    counts = {
        "tracks": 0,
        "matches": 0,
        "global_vehicles": 0,
        "trajectories": 0,
        "predictions": 0,
    }

    try:
        if payload.tracks and payload.camera_id:
            counts["tracks"] = ingest_tracks(db, payload.camera_id, payload.tracks, commit=False)
        if payload.matches:
            counts["matches"] = ingest_matches(db, payload.matches, commit=False)
        if payload.global_vehicles:
            counts["global_vehicles"] = ingest_global_vehicles(db, payload.global_vehicles, commit=False)
        if payload.trajectories:
            counts["trajectories"] = ingest_trajectories(db, payload.trajectories, commit=False)
        if payload.predictions:
            counts["predictions"] = ingest_predictions(db, payload.predictions, commit=False)
        db.commit()
        return counts
    except Exception:
        db.rollback()
        raise
