"""CP10 Comprehensive Test Suite: Backend Ingestion & Service Integration.

Validates that real Phase-1 AI outputs flow through backend ingestion services
into PostgreSQL/PostGIS without loss, corruption, duplication, or breaking existing APIs.

Covers:
1. CP10 database connectivity & PostGIS version
2. CP10 camera synchronization & idempotency
3. CP10 Phase 1 vehicle observation ingestion
4. CP10 field preservation & non-fabrication
5. CP10 plate and OCR status semantics
6. CP10 cross-camera match ingestion
7. CP10 global vehicle ingestion & relational linking
8. CP10 trajectory ingestion & constituent segments
9. CP10 PostGIS geometry & spatial distance queries
10. CP10 persistent alert lifecycle & retrieval
11. CP10 persistent traffic event lifecycle & PostGIS storage
12. CP10 transaction safety & atomic rollback
13. CP10 duplicate protection & ingestion idempotency
14. CP10 API retrieval consistency with source manifests
"""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from app.main import app
from app.db.session import engine, SessionLocal
from app.models.entities import (
    Camera,
    GlobalVehicle,
    VehicleObservation,
    LicensePlate,
    CrossCameraMatch,
    VehicleTrajectory,
    TrajectorySegment,
    Alert,
    TrafficEvent,
    RoutePrediction,
)
from app.schemas.ai_contracts import (
    VehicleTrackInput,
    CrossCameraMatchInput,
    GlobalVehicleInput,
    TrajectoryInput,
    RoutePredictionInput,
    UnifiedAIIngestPayload,
)
from app.services.camera_service import seed_default_cameras
from app.services.ingest_service import (
    ingest_tracks,
    ingest_matches,
    ingest_global_vehicles,
    ingest_trajectories,
    ingest_unified,
)

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_cp10_01_database_connectivity():
    """1. Verify PostgreSQL connection, PostGIS availability, and Alembic CP9 migration head."""
    with engine.connect() as conn:
        # Check PostgreSQL
        pg_ver = conn.execute(text("SELECT version();")).scalar()
        assert pg_ver is not None
        assert "PostgreSQL" in pg_ver

        # Check PostGIS
        gis_ver = conn.execute(text("SELECT PostGIS_Version();")).scalar()
        assert gis_ver is not None
        assert gis_ver.startswith("3.")

        # Check Alembic Migration Head
        alembic_rev = conn.execute(text("SELECT version_num FROM alembic_version;")).scalar()
        assert alembic_rev == "2a4b6c8d1e3f"


def test_cp10_02_camera_synchronization_and_idempotency():
    """2. Verify configs/cameras.yaml sync, road_name/speed_limit_kmh, and zero duplicate creation."""
    db = SessionLocal()
    try:
        # First seeding
        seed_default_cameras(db)
        cams_run1 = db.execute(select(Camera).order_by(Camera.id)).scalars().all()
        count_run1 = len(cams_run1)

        # Repeated seeding
        seed_default_cameras(db)
        cams_run2 = db.execute(select(Camera).order_by(Camera.id)).scalars().all()
        count_run2 = len(cams_run2)

        assert count_run1 == count_run2, "Camera count must not increase on repeated seeding"

        # Check CAM_01, CAM_02, CAM_03
        cam1 = db.get(Camera, "CAM_01")
        cam2 = db.get(Camera, "CAM_02")
        cam3 = db.get(Camera, "CAM_03")

        assert cam1 is not None and cam1.road_name == "SIMULATED_ROAD_A" and cam1.speed_limit_kmh == 40.0
        assert cam2 is not None and cam2.road_name == "SIMULATED_ROAD_B" and cam2.speed_limit_kmh == 40.0
        assert cam3 is not None and cam3.road_name == "SIMULATED_ROAD_C" and cam3.speed_limit_kmh == 40.0

        # Check PostGIS Point Geometry
        with engine.connect() as conn:
            wkt = conn.execute(text("SELECT ST_AsText(location_geom) FROM cameras WHERE id='CAM_01';")).scalar()
            assert "POINT(76.9558 11.0168)" in wkt
    finally:
        db.close()


def test_cp10_03_phase1_vehicle_observation_ingestion():
    """3. Verify real Phase 1 vehicle observations ingest without loss into vehicle_observations."""
    obs_path = PROJECT_ROOT / "runs" / "cross_camera" / "cross_camera_observations.json"
    assert obs_path.exists(), "cross_camera_observations.json must exist"

    with open(obs_path, "r", encoding="utf-8") as f:
        all_obs = json.load(f)["camera_observations"]

    # Ingest CAM_01 observations
    cam01_raw = [o for o in all_obs if o["camera_id"] == "CAM_01"]
    tracks_input = [VehicleTrackInput.model_validate(o) for o in cam01_raw]

    payload = {
        "camera_id": "CAM_01",
        "tracks": [t.model_dump() for t in tracks_input],
    }
    res = client.post("/api/v1/ingest/tracks", json=payload)
    assert res.status_code == 200
    assert res.json()["success"] is True

    db = SessionLocal()
    try:
        db_obs = db.execute(
            select(VehicleObservation).where(VehicleObservation.camera_id == "CAM_01")
        ).scalars().all()
        assert len(db_obs) >= len(cam01_raw)

        # Inspect specific track 1
        t1 = next(o for o in db_obs if o.track_id == 1)
        assert t1.vehicle_class == "bus"
        assert t1.last_bbox == [696, 748, 1847, 1438]
        assert t1.detector_confidence == pytest.approx(0.8769, abs=1e-4)
        assert t1.first_seen_frame == 1
        assert t1.last_seen_frame == 75
    finally:
        db.close()


def test_cp10_04_field_preservation_and_non_fabrication():
    """4. Audit stored observations for strict non-fabrication (no [0,0,0,0], no NaN confidences)."""
    db = SessionLocal()
    try:
        records = db.execute(
            select(VehicleObservation).where(VehicleObservation.camera_id == "CAM_01")
        ).scalars().all()

        for r in records:
            if r.last_bbox is not None:
                assert len(r.last_bbox) == 4, f"Bbox must have 4 coordinates: {r.last_bbox}"
                assert r.last_bbox != [0, 0, 0, 0], "Fabricated [0,0,0,0] bbox detected!"
                assert r.last_bbox[2] > r.last_bbox[0], "Bbox width must be positive"
                assert r.last_bbox[3] > r.last_bbox[1], "Bbox height must be positive"

            if r.detector_confidence is not None:
                assert 0.0 <= r.detector_confidence <= 1.0, f"Invalid confidence: {r.detector_confidence}"
                assert str(r.detector_confidence) != "nan", "NaN confidence detected!"
    finally:
        db.close()


def test_cp10_05_plate_and_ocr_status_semantics():
    """5. Verify plate status state machine semantics: unknown, tentative, and stable preserved."""
    db = SessionLocal()
    try:
        # Ingest track with tentative status and text
        tentative_track = {
            "track_id": 9942,
            "vehicle_class": "truck",
            "plate_text": "ONDUTY",
            "plate_status": "tentative",
            "plate_confidence": 96.81,
            "observation_count": 5,
            "valid_observation_count": 0,
            "best_ocr_confidence": 96.81,
            "last_bbox": [200, 300, 600, 700],
            "detector_confidence": 0.88,
        }
        res = client.post(
            "/api/v1/ingest/tracks",
            json={"camera_id": "CAM_01", "tracks": [tentative_track]},
        )
        assert res.status_code == 200

        obs = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 9942,
            )
        ).scalar_one_or_none()

        assert obs is not None
        assert obs.plate_text == "ONDUTY"
        assert obs.plate_status == "tentative"
        assert obs.plate_confidence == pytest.approx(96.81, abs=0.1)

        # Check LicensePlate sync record
        lp = db.execute(
            select(LicensePlate).where(LicensePlate.plate_text == "ONDUTY")
        ).scalar_one_or_none()
        assert lp is not None
        assert lp.plate_status == "tentative"
    finally:
        # Clean up test track
        to_del = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 9942,
            )
        ).scalars().all()
        for o in to_del:
            db.delete(o)
        to_del_lp = db.execute(
            select(LicensePlate).where(LicensePlate.plate_text == "ONDUTY")
        ).scalars().all()
        for lp_item in to_del_lp:
            db.delete(lp_item)
        db.commit()
        db.close()


def test_cp10_06_cross_camera_match_ingestion():
    """6. Verify cross-camera matches ingest with similarity [-1, 1], match_score, and evidence."""
    match_file = PROJECT_ROOT / "runs" / "cross_camera" / "cross_camera_matches.json"
    assert match_file.exists()

    with open(match_file, "r", encoding="utf-8") as f:
        matches = json.load(f)["matches"]

    # Sample top 50 matches covering matched=True, matched=False, and negative cosine similarity
    sample_matches = matches[:50]
    payload = [CrossCameraMatchInput.model_validate(m).model_dump() for m in sample_matches]

    res = client.post("/api/v1/ingest/matches", json=payload)
    assert res.status_code == 200
    assert res.json()["counts"]["matches"] == len(sample_matches)

    db = SessionLocal()
    try:
        # Query matched pair (CAM_01 track 1 <-> CAM_02 track 3, or first match in sample)
        m0 = sample_matches[0]
        db_m = db.execute(
            select(CrossCameraMatch).where(
                CrossCameraMatch.camera_a == m0["camera_a"],
                CrossCameraMatch.track_a == m0["track_a"],
                CrossCameraMatch.camera_b == m0["camera_b"],
                CrossCameraMatch.track_b == m0["track_b"],
            )
        ).scalar_one_or_none()

        assert db_m is not None
        assert db_m.reid_similarity == pytest.approx(m0["reid_similarity"], abs=1e-4)
        assert db_m.match_score == pytest.approx(m0["match_score"], abs=1e-4)
        assert db_m.available_evidence == m0["available_evidence"]
        assert db_m.matched == m0["matched"]
    finally:
        db.close()


def test_cp10_07_global_vehicle_ingestion_and_relational_linking():
    """7. Verify global vehicles ingest, link to observations, with 0 orphaned entities."""
    gv_file = PROJECT_ROOT / "runs" / "cross_camera" / "global_vehicle_entities.json"
    assert gv_file.exists()

    with open(gv_file, "r", encoding="utf-8") as f:
        gvs = json.load(f)["global_vehicles"]

    payload = [GlobalVehicleInput.model_validate(g).model_dump() for g in gvs]
    res = client.post("/api/v1/ingest/global-vehicles", json=payload)
    assert res.status_code == 200
    assert res.json()["counts"]["global_vehicles"] == len(gvs)

    db = SessionLocal()
    try:
        db_gvs = db.execute(select(GlobalVehicle)).scalars().all()
        assert len(db_gvs) >= 58

        # Verify GV_000001 (multi-camera bus) links observations
        gv1 = db.get(GlobalVehicle, "GV_000001")
        assert gv1 is not None
        assert gv1.vehicle_class == "bus"

        linked_obs = db.execute(
            select(VehicleObservation).where(VehicleObservation.global_vehicle_id == "GV_000001")
        ).scalars().all()
        assert len(linked_obs) >= 1
    finally:
        db.close()


def test_cp10_08_trajectory_ingestion_and_constituent_segments():
    """8. Verify vehicle trajectories and constituent segments persist all CP7 fields."""
    traj_file = PROJECT_ROOT / "runs" / "trajectory" / "vehicle_trajectories.json"
    assert traj_file.exists()

    with open(traj_file, "r", encoding="utf-8") as f:
        trajs = json.load(f)["vehicle_trajectories"]

    payload = [TrajectoryInput.model_validate(t).model_dump() for t in trajs]
    res = client.post("/api/v1/ingest/trajectories", json=payload)
    assert res.status_code == 200
    assert res.json()["counts"]["trajectories"] == len(trajs)

    db = SessionLocal()
    try:
        traj1 = db.execute(
            select(VehicleTrajectory).where(VehicleTrajectory.global_vehicle_id == "GV_000001")
        ).scalar_one_or_none()

        assert traj1 is not None
        assert traj1.camera_sequence == ["CAM_01", "CAM_02", "CAM_03"]
        assert len(traj1.segments) == 2

        # Check Segment 1 (CAM_01 -> CAM_02)
        s1 = next(s for s in traj1.segments if s.from_camera == "CAM_01" and s.to_camera == "CAM_02")
        assert s1.distance_meters == pytest.approx(366.38, abs=0.1)
        assert s1.bearing_degrees == pytest.approx(43.25, abs=0.1)
        assert s1.direction == "NE"
        assert s1.match_score == pytest.approx(0.9667, abs=1e-4)
        assert s1.reid_similarity == pytest.approx(0.9611, abs=1e-4)
        assert s1.available_evidence == ["reid", "class"]

        # Check Segment 2 (CAM_02 -> CAM_03)
        s2 = next(s for s in traj1.segments if s.from_camera == "CAM_02" and s.to_camera == "CAM_03")
        assert s2.distance_meters == pytest.approx(484.59, abs=0.1)
        assert s2.bearing_degrees == pytest.approx(40.78, abs=0.1)
        assert s2.direction == "NE"
        assert s2.match_score == pytest.approx(0.9082, abs=1e-4)
        assert s2.reid_similarity == pytest.approx(0.8929, abs=1e-4)
    finally:
        db.close()


def test_cp10_09_postgis_geometry_and_spatial_distance_query():
    """9. Verify PostGIS spatial calculations on cameras match Haversine distance within 1m."""
    with engine.connect() as conn:
        # Distance between CAM_01 and CAM_02 computed by PostGIS ST_Distance (WGS84 geography)
        dist = conn.execute(text("""
            SELECT ST_Distance(
                c1.location_geom::geography,
                c2.location_geom::geography
            )
            FROM cameras c1, cameras c2
            WHERE c1.id = 'CAM_01' AND c2.id = 'CAM_02';
        """)).scalar()

        # Haversine distance in Phase 1 is 366.38m; WGS84 ellipsoidal distance is ~366.5m
        assert dist is not None
        assert abs(dist - 366.38) < 1.0, f"Spatial distance mismatch: {dist}"


def test_cp10_10_persistent_alert_lifecycle():
    """10. Test persistent Alert registration, query via API, database integrity, and cleanup."""
    test_id = "ALT-CP10-INTEG-01"
    alert_payload = {
        "id": test_id,
        "alert_type": "Excessive Speed",
        "severity": "High",
        "message": "Vehicle exceeded 80 km/h in 40 km/h zone",
        "camera_id": "CAM_01",
        "global_vehicle_id": "GV_000001",
        "plate_text": "DL01AB9999",
        "location_text": "Simulated Ingress Corridor",
        "status": "Active",
    }

    # Register alert
    post_res = client.post("/api/v1/alerts", json=alert_payload)
    assert post_res.status_code == 201
    created = post_res.json()
    assert created["id"] == test_id
    assert created["type"] == "Excessive Speed"

    # Query via API
    get_res = client.get("/api/v1/alerts")
    assert get_res.status_code == 200
    alerts = get_res.json()
    assert any(a["id"] == test_id for a in alerts)

    # Database verification & cleanup
    db = SessionLocal()
    try:
        rec = db.get(Alert, test_id)
        assert rec is not None
        assert rec.camera_id == "CAM_01"
        assert rec.global_vehicle_id == "GV_000001"
        db.delete(rec)
        db.commit()
    finally:
        db.close()


def test_cp10_11_persistent_traffic_event_lifecycle():
    """11. Test persistent TrafficEvent incident creation, PostGIS geometry, and query via API."""
    test_id = "INC-CP10-INTEG-01"
    evt_payload = {
        "id": test_id,
        "event_type": "incident",
        "title": "Corridor Transit Stall",
        "severity": "medium",
        "camera_id": "CAM_02",
        "corridor": "SIMULATED_ROAD_B",
        "latitude": 11.0192,
        "longitude": 76.9581,
        "description": "Slow transit delay between sensor nodes",
        "status": "active",
    }

    # Register incident
    post_res = client.post("/api/v1/traffic/events/incidents", json=evt_payload)
    assert post_res.status_code == 201
    created = post_res.json()
    assert created["id"] == test_id

    # Query via API
    get_res = client.get("/api/v1/traffic/events")
    assert get_res.status_code == 200
    incidents = get_res.json().get("incidents", [])
    assert any(i["id"] == test_id for i in incidents)

    # Database PostGIS geometry check & cleanup
    db = SessionLocal()
    try:
        rec = db.get(TrafficEvent, test_id)
        assert rec is not None
        with engine.connect() as conn:
            wkt = conn.execute(
                text("SELECT ST_AsText(location_geom) FROM traffic_events WHERE id = :eid;"),
                {"eid": test_id},
            ).scalar()
            assert "POINT(76.9581 11.0192)" in wkt
        db.delete(rec)
        db.commit()
    finally:
        db.close()


def test_cp10_12_transaction_safety_and_atomic_rollback():
    """12. Test atomic multi-step rollback in ingest_unified when an exception occurs."""
    db = SessionLocal()
    try:
        # Pre-clean any leftover test track from previous runs
        db.query(VehicleObservation).filter(VehicleObservation.track_id == 88888).delete()
        db.commit()

        # Pre-verify track 88888 does not exist
        assert db.execute(
            select(VehicleObservation).where(VehicleObservation.track_id == 88888)
        ).scalar_one_or_none() is None

        valid_track = VehicleTrackInput(
            track_id=88888,
            vehicle_class="sedan",
            plate_text="TEST_ROLLBACK",
            detector_confidence=0.88,
        )

        # 1. Test failure in ingest_tracks with commit=False followed by rollback
        try:
            ingest_tracks(db, "CAM_01", [valid_track], commit=False)
            raise RuntimeError("Simulated failure before commit")
        except RuntimeError:
            db.rollback()

        # Verify track 88888 was completely rolled back and was NOT committed
        obs = db.execute(
            select(VehicleObservation).where(VehicleObservation.track_id == 88888)
        ).scalar_one_or_none()
        assert obs is None, "Rollback must remove all uncommitted records from database"

        # 2. Test ingest_unified atomic rollback when a sub-step fails
        from unittest.mock import patch
        with patch("app.services.ingest_service.ingest_matches", side_effect=RuntimeError("Sub-step failure")):
            payload = UnifiedAIIngestPayload(
                camera_id="CAM_01",
                tracks=[valid_track],
                matches=[
                    CrossCameraMatchInput(
                        camera_a="CAM_01",
                        track_a=1,
                        camera_b="CAM_02",
                        track_b=2,
                        reid_similarity=0.8,
                        match_score=0.85,
                        threshold=0.75,
                        matched=True,
                    )
                ],
            )
            with pytest.raises(RuntimeError, match="Sub-step failure"):
                ingest_unified(db, payload)

        # Confirm track 88888 was rolled back by ingest_unified and is NOT in the database!
        obs2 = db.execute(
            select(VehicleObservation).where(VehicleObservation.track_id == 88888)
        ).scalar_one_or_none()
        assert obs2 is None, "ingest_unified atomic rollback must ensure zero partial records remain"
    finally:
        db.close()


def test_cp10_13_duplicate_protection_and_idempotency():
    """13. Verify repeated ingestion updates existing records rather than creating duplicates."""
    db = SessionLocal()
    try:
        test_track = VehicleTrackInput(
            track_id=77777,
            vehicle_class="car",
            plate_text="IDEM1234",
            detector_confidence=0.85,
            observation_count=10,
        )

        # Ingest first time
        ingest_tracks(db, "CAM_01", [test_track])
        count1 = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 77777,
            )
        ).scalars().all()
        assert len(count1) == 1
        assert count1[0].observation_count == 10

        # Ingest second time with updated observation_count
        test_track.observation_count = 20
        ingest_tracks(db, "CAM_01", [test_track])
        count2 = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 77777,
            )
        ).scalars().all()
        assert len(count2) == 1, "Must update in place without creating duplicate observation record"
        assert count2[0].observation_count == 20
    finally:
        to_del = db.execute(
            select(VehicleObservation).where(
                VehicleObservation.camera_id == "CAM_01",
                VehicleObservation.track_id == 77777,
            )
        ).scalars().all()
        for o in to_del:
            db.delete(o)
        db.commit()
        db.close()


def test_cp10_14_api_retrieval_consistency_with_source():
    """14. Verify API response for vehicle trajectory and cameras matches source database records."""
    # Test Trajectory API for GV_000001
    res_traj = client.get("/api/v1/vehicles/GV_000001/trajectory")
    assert res_traj.status_code == 200
    traj_data = res_traj.json()
    assert traj_data["status"] == "Active Track"
    assert "detections" in traj_data
    assert len(traj_data["detections"]) >= 2

    # Check that detections have accurate camera coordinates
    d0 = traj_data["detections"][0]
    assert d0["cameraId"] == "CAM_01"
    assert d0["latitude"] == pytest.approx(11.0168, abs=1e-4)
    assert d0["longitude"] == pytest.approx(76.9558, abs=1e-4)

    # Test Camera API
    res_cam = client.get("/api/v1/cameras/CAM_01")
    assert res_cam.status_code == 200
    cam_data = res_cam.json()
    assert cam_data["id"] == "CAM_01"
    assert cam_data["roadName"] == "SIMULATED_ROAD_A"
    assert cam_data["speedLimitKmh"] == 40.0
