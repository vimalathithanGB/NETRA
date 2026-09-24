"""Database and PostGIS integration tests."""
from sqlalchemy import text, select
from app.db.session import SessionLocal, engine
from app.models.entities import Camera


def test_postgres_connection():
    """Verify raw PostgreSQL connection to netra_db."""
    with engine.connect() as conn:
        res = conn.execute(text("SELECT 1;"))
        assert res.scalar() == 1


def test_postgis_version():
    """Verify PostGIS extension is installed and responsive."""
    with engine.connect() as conn:
        version = conn.execute(text("SELECT PostGIS_Version();")).scalar()
        assert version is not None
        assert "3." in version


def test_camera_spatial_geometry():
    """Verify storing and querying spatial Point geometry with PostGIS."""
    db = SessionLocal()
    test_cam_id = "CAM-TEST-GEO-99"
    try:
        # Create camera with PostGIS WKT Point
        cam = Camera(
            id=test_cam_id,
            name="Spatial Calibration Node",
            latitude=28.6139,
            longitude=77.2090,
            location_geom="SRID=4326;POINT(77.2090 28.6139)",
            status="online",
            fps=60,
        )
        db.add(cam)
        db.commit()

        # Query using PostGIS spatial functions
        with engine.connect() as conn:
            stmt = text("SELECT ST_AsText(location_geom) FROM cameras WHERE id = :cid;")
            result = conn.execute(stmt, {"cid": test_cam_id}).scalar()
            assert result == "POINT(77.209 28.6139)"
    finally:
        # Cleanup
        to_del = db.get(Camera, test_cam_id)
        if to_del:
            db.delete(to_del)
            db.commit()
        db.close()
