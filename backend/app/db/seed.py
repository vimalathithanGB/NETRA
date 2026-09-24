"""Database Seeding Script for NETRA Development & Demo.

Populates initial cameras and representative multi-camera vehicle trajectories
matching the confirmed AI data contract and frontend presets.
"""
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.services.camera_service import seed_default_cameras
from app.services.ingest_service import ingest_trajectories
from app.schemas.ai_contracts import TrajectoryInput, TrajectorySegmentInput

SEED_TRAJECTORIES = [
    TrajectoryInput(
        global_vehicle_id="GV-TN-38-AB1234",
        vehicle_class="Car",
        plate_text="TN38AB1234",
        camera_sequence=["CAM-001", "CAM-004", "CAM-007", "CAM-012"],
        total_distance_meters=14800.0,
        total_travel_time_seconds=720.0,
        average_speed_kmh=52.4,  # Simulation artifact
        segments=[
            TrajectorySegmentInput(
                from_camera="CAM-001",
                to_camera="CAM-004",
                from_timestamp="14:32:10",
                to_timestamp="14:36:42",
                distance_meters=5500.0,
                cumulative_distance_meters=5500.0,
                travel_time_seconds=272.0,
                average_speed_kmh=58.0,
                speed_limit_kmh=60.0,
                minimum_feasible_time_seconds=200.0,
                travel_time_feasible=True,
                bearing_degrees=135.0,
                direction="South-East",
            ),
            TrajectorySegmentInput(
                from_camera="CAM-004",
                to_camera="CAM-007",
                from_timestamp="14:36:42",
                to_timestamp="14:40:18",
                distance_meters=4100.0,
                cumulative_distance_meters=9600.0,
                travel_time_seconds=216.0,
                average_speed_kmh=51.0,
                speed_limit_kmh=50.0,
                minimum_feasible_time_seconds=180.0,
                travel_time_feasible=True,
                bearing_degrees=45.0,
                direction="North-East",
            ),
            TrajectorySegmentInput(
                from_camera="CAM-007",
                to_camera="CAM-012",
                from_timestamp="14:40:18",
                to_timestamp="14:44:03",
                distance_meters=5200.0,
                cumulative_distance_meters=14800.0,
                travel_time_seconds=225.0,
                average_speed_kmh=56.0,
                speed_limit_kmh=60.0,
                minimum_feasible_time_seconds=200.0,
                travel_time_feasible=True,
                bearing_degrees=90.0,
                direction="East",
            ),
        ],
    ),
    TrajectoryInput(
        global_vehicle_id="GV-DL-04-C8891",
        vehicle_class="Commercial SUV",
        plate_text="DL 04 C 8891",
        camera_sequence=["CAM-005", "CAM-003", "CAM-004", "CAM-001"],
        total_distance_meters=18200.0,
        total_travel_time_seconds=1448.0,
        average_speed_kmh=58.2,
        segments=[
            TrajectorySegmentInput(
                from_camera="CAM-005",
                to_camera="CAM-003",
                from_timestamp="13:58:00",
                to_timestamp="14:08:14",
                distance_meters=8500.0,
                cumulative_distance_meters=8500.0,
                travel_time_seconds=614.0,
                average_speed_kmh=50.0,
                speed_limit_kmh=60.0,
                minimum_feasible_time_seconds=400.0,
                travel_time_feasible=True,
                bearing_degrees=35.0,
                direction="North-East",
            ),
            TrajectorySegmentInput(
                from_camera="CAM-003",
                to_camera="CAM-004",
                from_timestamp="14:08:14",
                to_timestamp="14:15:30",
                distance_meters=4700.0,
                cumulative_distance_meters=13200.0,
                travel_time_seconds=436.0,
                average_speed_kmh=55.0,
                speed_limit_kmh=60.0,
                minimum_feasible_time_seconds=300.0,
                travel_time_feasible=True,
                bearing_degrees=0.0,
                direction="North",
            ),
            TrajectorySegmentInput(
                from_camera="CAM-004",
                to_camera="CAM-001",
                from_timestamp="14:15:30",
                to_timestamp="14:22:08",
                distance_meters=5000.0,
                cumulative_distance_meters=18200.0,
                travel_time_seconds=398.0,
                average_speed_kmh=52.0,
                speed_limit_kmh=60.0,
                minimum_feasible_time_seconds=250.0,
                travel_time_feasible=True,
                bearing_degrees=0.0,
                direction="North",
            ),
        ],
    ),
    TrajectoryInput(
        global_vehicle_id="GV-HR-26-DK5092",
        vehicle_class="Sedan",
        plate_text="HR 26 DK 5092",
        camera_sequence=["CAM-002", "CAM-007", "CAM-012"],
        total_distance_meters=12100.0,
        total_travel_time_seconds=607.0,
        average_speed_kmh=46.0,
        segments=[
            TrajectorySegmentInput(
                from_camera="CAM-002",
                to_camera="CAM-007",
                from_timestamp="14:12:00",
                to_timestamp="14:18:22",
                distance_meters=5900.0,
                cumulative_distance_meters=5900.0,
                travel_time_seconds=382.0,
                average_speed_kmh=48.0,
                speed_limit_kmh=50.0,
                minimum_feasible_time_seconds=250.0,
                travel_time_feasible=True,
                bearing_degrees=140.0,
                direction="South-East",
            ),
            TrajectorySegmentInput(
                from_camera="CAM-007",
                to_camera="CAM-012",
                from_timestamp="14:18:22",
                to_timestamp="14:22:07",
                distance_meters=6200.0,
                cumulative_distance_meters=12100.0,
                travel_time_seconds=225.0,
                average_speed_kmh=52.0,
                speed_limit_kmh=60.0,
                minimum_feasible_time_seconds=200.0,
                travel_time_feasible=True,
                bearing_degrees=90.0,
                direction="East",
            ),
        ],
    ),
    TrajectoryInput(
        global_vehicle_id="GV-UP-16-F3301",
        vehicle_class="Heavy Truck",
        plate_text="UP 16 F 3301",
        camera_sequence=["CAM-006", "CAM-004", "CAM-012"],
        total_distance_meters=15600.0,
        total_travel_time_seconds=1195.0,
        average_speed_kmh=41.5,
        segments=[
            TrajectorySegmentInput(
                from_camera="CAM-006",
                to_camera="CAM-004",
                from_timestamp="14:02:10",
                to_timestamp="14:14:45",
                distance_meters=7800.0,
                cumulative_distance_meters=7800.0,
                travel_time_seconds=755.0,
                average_speed_kmh=42.0,
                speed_limit_kmh=50.0,
                minimum_feasible_time_seconds=450.0,
                travel_time_feasible=True,
                bearing_degrees=150.0,
                direction="South-East",
            ),
            TrajectorySegmentInput(
                from_camera="CAM-004",
                to_camera="CAM-012",
                from_timestamp="14:14:45",
                to_timestamp="14:22:05",
                distance_meters=7800.0,
                cumulative_distance_meters=15600.0,
                travel_time_seconds=440.0,
                average_speed_kmh=40.0,
                speed_limit_kmh=50.0,
                minimum_feasible_time_seconds=300.0,
                travel_time_feasible=True,
                bearing_degrees=90.0,
                direction="East",
            ),
        ],
    ),
]


def seed_all(db: Session) -> None:
    """Run full seed for cameras and trajectory records."""
    print("Seeding default cameras...")
    cam_count = seed_default_cameras(db)
    print(f"Seeded {cam_count} new cameras.")

    print("Seeding representative trajectories...")
    traj_count = ingest_trajectories(db, SEED_TRAJECTORIES)
    print(f"Seeded {traj_count} trajectories.")


if __name__ == "__main__":
    db = SessionLocal()
    try:
        seed_all(db)
        print("Database seeding completed successfully.")
    finally:
        db.close()
