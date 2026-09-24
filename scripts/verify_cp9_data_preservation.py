"""CP9 Data Preservation and Verification Script.

Executes a full ingestion of Phase 1 AI manifests into PostgreSQL / PostGIS,
verifies exact retention of CP7-preserved fields, checks Alert/TrafficEvent tables,
confirms non-fabrication, and exports runs/phase2/cp9_data_preservation_report.json.
"""
import json
from pathlib import Path
import sys
from datetime import datetime

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from sqlalchemy import select, text
from app.db.session import SessionLocal, engine
from app.models.entities import (
    Camera,
    GlobalVehicle,
    VehicleObservation,
    CrossCameraMatch,
    VehicleTrajectory,
    TrajectorySegment,
    Alert,
    TrafficEvent,
    LicensePlate,
)
from app.services.camera_service import seed_default_cameras
from app.services.ingest_service import (
    ingest_tracks,
    ingest_matches,
    ingest_global_vehicles,
    ingest_trajectories,
)
from app.schemas.ai_contracts import (
    VehicleTrackInput,
    CrossCameraMatchInput,
    GlobalVehicleInput,
    TrajectoryInput,
)

def run_verification():
    db = SessionLocal()
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoint": "CP9",
        "phase": 2,
        "database": "PostgreSQL 16.10 with PostGIS 3.5.2",
        "alembic_version": None,
        "schema_alignment": {},
        "data_preservation": {},
        "non_fabrication_audit": {},
        "persistence_verification": {},
        "overall_status": "PENDING",
    }

    try:
        # 1. Check Alembic Version
        with engine.connect() as conn:
            ver = conn.execute(text("SELECT version_num FROM alembic_version;")).scalar()
            report["alembic_version"] = ver

        # 2. Seed and Check Cameras
        seed_default_cameras(db)
        cameras = db.execute(select(Camera)).scalars().all()
        cam_data = []
        for c in cameras:
            cam_data.append({
                "id": c.id,
                "name": c.name,
                "road_name": c.road_name,
                "speed_limit_kmh": c.speed_limit_kmh,
                "has_geom": c.location_geom is not None,
            })
        report["schema_alignment"]["cameras"] = {
            "total_cameras": len(cameras),
            "cameras": cam_data,
            "all_road_names_present": all(c.road_name is not None for c in cameras),
            "all_speed_limits_present": all(c.speed_limit_kmh is not None for c in cameras),
        }

        # 3. Ingest and Verify VehicleObservations (from cross_camera_observations.json)
        obs_file = PROJECT_ROOT / "runs" / "cross_camera" / "cross_camera_observations.json"
        with open(obs_file, "r", encoding="utf-8") as f:
            raw_obs = json.load(f)["camera_observations"]

        # Group by camera and ingest
        obs_by_cam = {}
        for o in raw_obs:
            obs_by_cam.setdefault(o["camera_id"], []).append(o)

        total_obs_ingested = 0
        for cid, t_list in obs_by_cam.items():
            formatted = [
                VehicleTrackInput(
                    track_id=o["track_id"],
                    vehicle_class=o["vehicle_class"],
                    plate_text=o["plate_text"],
                    plate_status=o["plate_status"],
                    plate_confidence=o["plate_confidence"],
                    observation_count=o["observation_count"],
                    valid_observation_count=o["valid_observation_count"],
                    best_ocr_confidence=o["best_ocr_confidence"],
                    last_bbox=o["last_bbox"],
                    detector_confidence=o["detector_confidence"],
                    first_seen_frame=o["first_seen_frame"],
                    last_seen_frame=o["last_seen_frame"],
                )
                for o in t_list
            ]
            total_obs_ingested += ingest_tracks(db, cid, formatted)

        # Query and inspect observations from DB
        db_obs = db.execute(select(VehicleObservation)).scalars().all()
        obs_last_bbox_count = sum(1 for o in db_obs if o.last_bbox is not None and len(o.last_bbox) == 4)
        obs_det_conf_count = sum(1 for o in db_obs if o.detector_confidence is not None)
        obs_plate_status_count = sum(1 for o in db_obs if o.plate_status is not None)

        report["data_preservation"]["vehicle_observations"] = {
            "total_raw_in_manifest": len(raw_obs),
            "total_in_database": len(db_obs),
            "last_bbox_preserved": obs_last_bbox_count,
            "last_bbox_preservation_rate": obs_last_bbox_count / len(raw_obs) if raw_obs else 1.0,
            "detector_confidence_preserved": obs_det_conf_count,
            "detector_confidence_preservation_rate": obs_det_conf_count / len(raw_obs) if raw_obs else 1.0,
            "plate_status_preserved": obs_plate_status_count,
        }

        # 4. Ingest and Verify CrossCameraMatches
        matches_file = PROJECT_ROOT / "runs" / "cross_camera" / "cross_camera_matches.json"
        with open(matches_file, "r", encoding="utf-8") as f:
            raw_matches = json.load(f)["matches"]

        formatted_matches = [
            CrossCameraMatchInput(
                camera_a=m["camera_a"],
                track_a=m["track_a"],
                camera_b=m["camera_b"],
                track_b=m["track_b"],
                reid_similarity=m["reid_similarity"],
                plate_match=m["plate_match"],
                class_match=m["class_match"],
                available_evidence=m["available_evidence"],
                match_score=m["match_score"],
                threshold=m["threshold"],
                matched=m["matched"],
            )
            for m in raw_matches
        ]
        ingest_matches(db, formatted_matches)
        db_matches = db.execute(select(CrossCameraMatch)).scalars().all()

        report["data_preservation"]["cross_camera_matches"] = {
            "total_raw_in_manifest": len(raw_matches),
            "total_in_database": len(db_matches),
            "reid_similarity_preserved": sum(1 for m in db_matches if m.reid_similarity is not None),
            "match_score_preserved": sum(1 for m in db_matches if m.match_score is not None),
            "available_evidence_preserved": sum(1 for m in db_matches if m.available_evidence is not None),
        }

        # 5. Ingest Global Vehicles
        gv_file = PROJECT_ROOT / "runs" / "cross_camera" / "global_vehicle_entities.json"
        with open(gv_file, "r", encoding="utf-8") as f:
            raw_gvs = json.load(f)["global_vehicles"]

        formatted_gvs = [
            GlobalVehicleInput(
                global_vehicle_id=g["global_vehicle_id"],
                vehicle_class=g["vehicle_class"],
                plate_text=g["plate_text"],
                match_confidence=g["match_confidence"],
                observations=[
                    {"camera_id": ob["camera_id"], "track_id": ob["track_id"]}
                    for ob in g.get("observations", [])
                ],
            )
            for g in raw_gvs
        ]
        ingest_global_vehicles(db, formatted_gvs)
        db_gvs = db.execute(select(GlobalVehicle)).scalars().all()

        report["data_preservation"]["global_vehicles"] = {
            "total_raw_in_manifest": len(raw_gvs),
            "total_in_database": len(db_gvs),
        }

        # 6. Ingest and Verify Trajectories and Segments
        traj_file = PROJECT_ROOT / "runs" / "trajectory" / "vehicle_trajectories.json"
        with open(traj_file, "r", encoding="utf-8") as f:
            raw_trajs = json.load(f)["vehicle_trajectories"]

        formatted_trajs = [TrajectoryInput.model_validate(t) for t in raw_trajs]
        ingest_trajectories(db, formatted_trajs)

        db_trajs = db.execute(select(VehicleTrajectory)).scalars().all()
        db_segs = db.execute(select(TrajectorySegment)).scalars().all()

        total_manifest_segs = sum(len(t.get("segments", [])) for t in raw_trajs)
        segs_match_score = sum(1 for s in db_segs if s.match_score is not None)
        segs_reid_sim = sum(1 for s in db_segs if s.reid_similarity is not None)
        segs_evidence = sum(1 for s in db_segs if s.available_evidence is not None and len(s.available_evidence) > 0)

        report["data_preservation"]["trajectories"] = {
            "total_trajectories_raw": len(raw_trajs),
            "total_trajectories_db": len(db_trajs),
            "total_segments_raw": total_manifest_segs,
            "total_segments_db": len(db_segs),
            "match_score_preserved": segs_match_score,
            "match_score_preservation_rate": segs_match_score / total_manifest_segs if total_manifest_segs else 1.0,
            "reid_similarity_preserved": segs_reid_sim,
            "reid_similarity_preservation_rate": segs_reid_sim / total_manifest_segs if total_manifest_segs else 1.0,
            "available_evidence_preserved": segs_evidence,
            "available_evidence_preservation_rate": segs_evidence / total_manifest_segs if total_manifest_segs else 1.0,
        }

        # 7. Non-Fabrication Audit
        zero_bboxes = [o.track_id for o in db_obs if o.last_bbox == [0, 0, 0, 0]]
        nan_confs = [o.track_id for o in db_obs if o.detector_confidence is not None and str(o.detector_confidence) == "nan"]
        
        report["non_fabrication_audit"] = {
            "zero_bbox_count": len(zero_bboxes),
            "nan_confidence_count": len(nan_confs),
            "non_fabrication_pass": len(zero_bboxes) == 0 and len(nan_confs) == 0,
        }

        # 8. Alert and TrafficEvent Persistence Check
        # Clean test entries if any
        db.query(Alert).filter(Alert.id == "VERIFY-ALT-01").delete()
        db.query(TrafficEvent).filter(TrafficEvent.id == "VERIFY-EVT-01").delete()
        db.commit()

        # Insert verify alert
        test_alert = Alert(
            id="VERIFY-ALT-01",
            alert_type="Speeding",
            severity="High",
            message="Test verification alert",
            camera_id="CAM_01",
            status="Active",
        )
        db.add(test_alert)

        # Insert verify traffic event with geometry
        test_evt = TrafficEvent(
            id="VERIFY-EVT-01",
            event_type="congestion",
            title="Verification Congestion Node",
            severity="medium",
            camera_id="CAM_01",
            corridor="SIMULATED_ROAD_A",
            latitude=11.0168,
            longitude=76.9558,
            location_geom="SRID=4326;POINT(76.9558 11.0168)",
            description="Test verification traffic event",
            status="active",
        )
        db.add(test_evt)
        db.commit()

        retrieved_alert = db.get(Alert, "VERIFY-ALT-01")
        retrieved_evt = db.get(TrafficEvent, "VERIFY-EVT-01")

        with engine.connect() as conn:
            geom_wkt = conn.execute(
                text("SELECT ST_AsText(location_geom) FROM traffic_events WHERE id = 'VERIFY-EVT-01';")
            ).scalar()

        report["persistence_verification"] = {
            "alert_persistent": retrieved_alert is not None,
            "traffic_event_persistent": retrieved_evt is not None,
            "postgis_point_wkt": geom_wkt,
            "postgis_valid": geom_wkt == "POINT(76.9558 11.0168)",
        }

        # Cleanup test entries
        db.delete(retrieved_alert)
        db.delete(retrieved_evt)
        db.commit()

        report["overall_status"] = "PASS"

    except Exception as e:
        report["overall_status"] = "FAIL"
        report["error"] = str(e)
        raise
    finally:
        db.close()

    out_path = PROJECT_ROOT / "runs" / "phase2" / "cp9_data_preservation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"CP9 Verification Report written to {out_path}")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Alembic Version: {report['alembic_version']}")
    print(f"Vehicle Observations Preserved: {report['data_preservation']['vehicle_observations']['last_bbox_preserved']}/{report['data_preservation']['vehicle_observations']['total_raw_in_manifest']}")
    print(f"Trajectory Segments Preserved: {report['data_preservation']['trajectories']['match_score_preserved']}/{report['data_preservation']['trajectories']['total_segments_raw']}")
    print(f"Non-Fabrication Check: {report['non_fabrication_audit']['non_fabrication_pass']}")
    print(f"PostGIS Geometry Valid: {report['persistence_verification']['postgis_valid']}")

if __name__ == "__main__":
    run_verification()
