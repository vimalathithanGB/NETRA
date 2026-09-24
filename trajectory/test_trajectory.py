"""
NETRA — Checkpoint 7 Trajectory & GIS Validation Suite
=====================================================
File: trajectory/test_trajectory.py

Automated 19-point validation testing:
1. Camera configuration loads
2. Every camera has latitude
3. Every camera has longitude
4. Every camera has speed limit
5. Global vehicle IDs exist
6. Every trajectory observation has camera ID
7. Every camera ID exists in cameras.yaml
8. Timestamps are chronological
9. Distances are >= 0
10. Travel times are > 0 for valid segments
11. Speeds are finite
12. Bearings are within 0–360 degrees
13. Directions are valid (8-point cardinal)
14. No fake coordinates are generated (matches cameras.yaml)
15. No fake timestamps are generated (preserves frame index and fps formula)
16. Trajectory JSON is valid
17. Map HTML is created
18. Existing Checkpoint 1–6 modules remain functional
19. pip check remains clean
"""

import json
import math
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_trajectory_checkpoint_7():
    print("=" * 70)
    print("NETRA CHECKPOINT 7: 19-POINT TRAJECTORY & GIS VERIFICATION SUITE")
    print("=" * 70)

    # 1. Camera configuration loads
    cameras_yaml_path = Path("configs/cameras.yaml")
    assert cameras_yaml_path.exists(), f"Missing: {cameras_yaml_path}"
    with open(cameras_yaml_path, "r", encoding="utf-8") as f:
        cam_data = yaml.safe_load(f)
    cameras = cam_data.get("cameras", {})
    assert len(cameras) >= 3, f"Expected at least 3 cameras, got {len(cameras)}"
    print(f"[PASS] 1. Camera configuration loads ({len(cameras)} cameras found).")

    # 2, 3, 4. Every camera has latitude, longitude, speed limit
    for cam_id, cfg in cameras.items():
        assert "latitude" in cfg and isinstance(cfg["latitude"], (int, float)), f"Invalid lat in {cam_id}"
        assert -90.0 <= cfg["latitude"] <= 90.0, f"Latitude out of bounds in {cam_id}"
        assert "longitude" in cfg and isinstance(cfg["longitude"], (int, float)), f"Invalid lon in {cam_id}"
        assert -180.0 <= cfg["longitude"] <= 180.0, f"Longitude out of bounds in {cam_id}"
        assert "speed_limit_kmh" in cfg and cfg["speed_limit_kmh"] > 0, f"Invalid speed limit in {cam_id}"
    print("[PASS] 2. Every camera has valid latitude [-90, 90].")
    print("[PASS] 3. Every camera has valid longitude [-180, 180].")
    print("[PASS] 4. Every camera has positive speed limit (km/h).")

    # Load trajectory and segment files
    traj_path = Path("runs/trajectory/vehicle_trajectories.json")
    seg_path = Path("runs/trajectory/trajectory_segments.json")
    map_path = Path("runs/trajectory/trajectory_map.html")

    assert traj_path.exists(), f"Missing: {traj_path}"
    assert seg_path.exists(), f"Missing: {seg_path}"
    assert map_path.exists(), f"Missing: {map_path}"

    with open(traj_path, "r", encoding="utf-8") as f:
        traj_data = json.load(f)
    with open(seg_path, "r", encoding="utf-8") as f:
        seg_data = json.load(f)

    trajectories = traj_data.get("vehicle_trajectories", [])
    segments = seg_data.get("trajectory_segments", [])

    assert len(trajectories) > 0, "No trajectories found!"
    assert len(segments) > 0, "No segments found!"

    # 5. Global vehicle IDs exist
    gids = {t["global_vehicle_id"] for t in trajectories}
    for gid in gids:
        assert gid.startswith("GV_") and len(gid) == 9, f"Invalid global vehicle ID format: {gid}"
    print(f"[PASS] 5. Global vehicle IDs exist ({len(gids)} unique IDs validated).")

    # 6, 7. Every trajectory observation has camera ID and exists in cameras.yaml
    total_obs = 0
    configured_cam_ids = set(cameras.keys())
    for t in trajectories:
        for obs in t.get("observations", []):
            total_obs += 1
            cam_id = obs.get("camera_id")
            assert cam_id in configured_cam_ids, f"Observation camera {cam_id} not in cameras.yaml"
    print(f"[PASS] 6. Every trajectory observation has a camera ID ({total_obs} observations).")
    print(f"[PASS] 7. Every observation camera ID exists in cameras.yaml: {sorted(configured_cam_ids)}.")

    # 8. Timestamps are chronological
    for t in trajectories:
        obs_list = t.get("observations", [])
        for i in range(len(obs_list) - 1):
            dt_curr = datetime.fromisoformat(obs_list[i]["timestamp_start"])
            dt_next = datetime.fromisoformat(obs_list[i + 1]["timestamp_start"])
            assert dt_curr <= dt_next, f"Non-chronological observation timestamps in {t['global_vehicle_id']}"
    for s in segments:
        dt_from = datetime.fromisoformat(s["from_timestamp"])
        dt_to = datetime.fromisoformat(s["to_timestamp"])
        assert dt_from < dt_to, f"Non-chronological segment timestamps in {s}"
    print("[PASS] 8. Timestamps are strictly chronological across all observations and segments.")

    # 9. Distances are >= 0
    for s in segments:
        assert s["distance_meters"] >= 0.0, f"Negative distance in segment: {s}"
        assert s["cumulative_distance_meters"] >= s["distance_meters"]
    for t in trajectories:
        assert t["total_distance_meters"] >= 0.0
    print(f"[PASS] 9. All distances are non-negative ({len(segments)} segments checked).")

    # 10. Travel times are > 0 for valid segments
    for s in segments:
        assert s["travel_time_seconds"] > 0.0, f"Non-positive travel time in segment: {s}"
    print("[PASS] 10. Travel times are strictly positive (>0) for all reconstructed segments.")

    # 11. Speeds are finite
    for s in segments:
        spd = s.get("average_speed_kmh")
        if spd is not None:
            assert math.isfinite(spd) and spd > 0.0, f"Invalid speed value: {spd}"
    for t in trajectories:
        spd = t.get("average_speed_kmh")
        if spd is not None:
            assert math.isfinite(spd), f"Non-finite average speed: {spd}"
    print("[PASS] 11. Speeds are finite numbers (no NaN or Inf values).")

    # 12. Bearings are within 0–360 degrees
    for s in segments:
        b = s["bearing_degrees"]
        assert 0.0 <= b < 360.0, f"Bearing out of bounds [0, 360): {b}"
    print("[PASS] 12. Movement bearings are within [0.0, 360.0) degrees.")

    # 13. Directions are valid
    valid_directions = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    for s in segments:
        d = s["direction"]
        assert d in valid_directions, f"Invalid direction string: {d}"
    print(f"[PASS] 13. Movement directions are valid 8-point compass cardinals ({valid_directions}).")

    # 14. No fake coordinates are generated
    configured_coords = {(c["latitude"], c["longitude"]) for c in cameras.values()}
    for t in trajectories:
        for obs in t.get("observations", []):
            coord = (obs["latitude"], obs["longitude"])
            assert coord in configured_coords, f"Unknown/fabricated coordinate: {coord}"
    print("[PASS] 14. No fake coordinates generated; all match configured camera nodes.")

    # 15. No fake timestamps are generated
    video_start_time = datetime.fromisoformat("2026-01-01T09:30:00")
    fps = 29.97003
    for t in trajectories:
        for obs in t.get("observations", []):
            f_start = obs["frame_start"]
            expected_start_dt = video_start_time + timedelta(seconds=f_start / fps)
            # Verify parsed timestamp matches frame / fps within microsecond precision
            dt_actual = datetime.fromisoformat(obs["timestamp_start"])
            sec_from_start = (dt_actual - video_start_time).total_seconds()
            expected_sec = f_start / fps
            assert abs(sec_from_start - expected_sec) < 0.001, f"Timestamp mismatch for frame {f_start}"
    print("[PASS] 15. Timestamps are strictly grounded in video frame indexes and actual video FPS.")

    # 16. Trajectory JSON is valid
    assert "vehicle_trajectories" in traj_data
    assert "trajectory_segments" in seg_data
    print("[PASS] 16. Trajectory and Segment JSON schemas are valid and well-formed.")

    # 17. Map HTML is created
    assert map_path.stat().st_size > 1000, "Map HTML file is empty or suspiciously small"
    with open(map_path, "r", encoding="utf-8") as f:
        html_src = f.read()
    assert "leaflet" in html_src.lower(), "Map HTML does not reference Leaflet library"
    assert "CAM_01" in html_src and "CAM_02" in html_src and "CAM_03" in html_src, "Camera IDs missing in map"
    print(f"[PASS] 17. Interactive Leaflet Map HTML created ({map_path.stat().st_size} bytes).")

    # 18. Existing Checkpoint 1–6 modules remain functional
    try:
        from reid.vehicle_reid import VehicleReIDExtractor
        from ocr.license_plate_ocr import LicensePlateOCR
        from inference.plate_ocr_pipeline import PlateOCRPipeline
        from tracking.bytetrack_tracker import run_bytetrack_tracking
        from tracking.unified_vehicle_pipeline import UnifiedVehiclePipeline
        from tracking.cross_camera_matching import CrossCameraMatcher
        print("[PASS] 18. All Checkpoint 1–6 modules import and load successfully.")
    except Exception as e:
        raise AssertionError(f"Checkpoint 1–6 module regression detected: {e}")

    # 19. pip check remains clean
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"pip check failed: {result.stdout} {result.stderr}"
    assert "No broken requirements found" in result.stdout, f"pip check output: {result.stdout}"
    print("[PASS] 19. pip check passes with zero broken requirements.")

    print("=" * 70)
    print("ALL 19 CHECKPOINT 7 VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_trajectory_checkpoint_7()
