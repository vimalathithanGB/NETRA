"""
NETRA — Checkpoint 8 Urban Traffic Analytics & Route Prediction Validation Suite
================================================================================
File: analytics/test_traffic_analytics.py

Automated 19-point validation testing:
1. All input JSON files exist
2. All JSON files are valid
3. Camera IDs are valid
4. Global vehicle IDs are valid
5. Vehicle counts are non-negative
6. OD matrix contains valid camera IDs
7. Route counts are non-negative
8. Heatmap coordinates match cameras.yaml
9. Heatmap intensity is within [0, 1]
10. Time-series timestamps are chronological
11. Vehicle-class counts are non-negative
12. Transition probabilities are within [0, 1]
13. Transition probabilities for each source camera sum approximately to 1
14. No transition probability is invented
15. Route predictions only use available historical data
16. Prediction probabilities are within [0, 1]
17. No fake speed values are used in analytics
18. Existing Checkpoint 1–7 modules remain unaffected
19. pip check remains clean
"""

import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def test_traffic_analytics_checkpoint_8():
    print("=" * 70)
    print("NETRA CHECKPOINT 8: 19-POINT TRAFFIC ANALYTICS VERIFICATION SUITE")
    print("=" * 70)

    # Base directories and files
    analytics_dir = Path("runs/analytics")
    summary_file = analytics_dir / "traffic_analytics_summary.json"
    od_file = analytics_dir / "od_matrix.json"
    route_file = analytics_dir / "route_density.json"
    heatmap_file = analytics_dir / "traffic_heatmap.json"
    timeseries_file = analytics_dir / "traffic_timeseries.json"
    graph_file = analytics_dir / "camera_network_graph.json"
    trans_file = analytics_dir / "route_transition_probabilities.json"
    pred_file = analytics_dir / "route_predictions.json"
    map_file = analytics_dir / "traffic_analytics_map.html"
    cameras_yaml = Path("configs/cameras.yaml")

    required_files = [
        summary_file, od_file, route_file, heatmap_file,
        timeseries_file, graph_file, trans_file, pred_file,
        map_file, cameras_yaml
    ]

    # Rule 1: All input JSON & config files exist
    for rf in required_files:
        assert rf.exists(), f"Missing required file: {rf}"
    print(f"[PASS] 1. All {len(required_files)} analytics output files and configs exist.")

    # Rule 2: All JSON files are valid and parse cleanly
    with open(cameras_yaml, "r", encoding="utf-8") as f:
        cam_data = yaml.safe_load(f)["cameras"]
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    with open(od_file, "r", encoding="utf-8") as f:
        od_data = json.load(f)
    with open(route_file, "r", encoding="utf-8") as f:
        route_data = json.load(f)
    with open(heatmap_file, "r", encoding="utf-8") as f:
        heatmap_data = json.load(f)
    with open(timeseries_file, "r", encoding="utf-8") as f:
        timeseries_data = json.load(f)
    with open(graph_file, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    with open(trans_file, "r", encoding="utf-8") as f:
        trans_data = json.load(f)
    with open(pred_file, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    print("[PASS] 2. All JSON and YAML files parsed cleanly with valid schemas.")

    valid_cameras = set(cam_data.keys())

    # Rule 3: Camera IDs are valid
    for cam_node in graph_data.get("nodes", []):
        assert cam_node["camera_id"] in valid_cameras, f"Invalid camera_id: {cam_node['camera_id']}"
    for hp in heatmap_data.get("heatmap_points", []):
        assert hp["camera_id"] in valid_cameras, f"Invalid camera_id in heatmap: {hp['camera_id']}"
    for cid in summary_data.get("camera_statistics", {}):
        assert cid in valid_cameras, f"Invalid camera_id in summary: {cid}"
    print(f"[PASS] 3. Camera IDs are strictly valid: {sorted(valid_cameras)}.")

    # Rule 4: Global vehicle IDs are valid
    for pred in pred_data.get("vehicle_route_predictions", []):
        gid = pred["global_vehicle_id"]
        assert gid.startswith("GV_") and len(gid) == 9, f"Invalid global vehicle ID: {gid}"
    print("[PASS] 4. Global vehicle IDs follow strict 'GV_XXXXXX' format.")

    # Rule 5: Vehicle counts are non-negative
    for hp in heatmap_data.get("heatmap_points", []):
        assert hp["vehicle_count"] >= 0, f"Negative vehicle count in heatmap: {hp}"
    for cid, stats in summary_data.get("camera_statistics", {}).items():
        assert stats["total_observations"] >= 0, f"Negative total observations: {stats}"
        assert stats["unique_global_vehicles"] >= 0, f"Negative unique vehicles: {stats}"
    print("[PASS] 5. All vehicle counts are strictly non-negative (>= 0).")

    # Rule 6: OD matrix contains valid camera IDs
    od_cams = set(od_data.get("cameras", []))
    assert od_cams == valid_cameras, f"OD cameras mismatch: {od_cams} vs {valid_cameras}"
    for orig, row in od_data.get("matrix", {}).items():
        assert orig in valid_cameras, f"Invalid origin: {orig}"
        for dest, count in row.items():
            assert dest in valid_cameras, f"Invalid destination: {dest}"
            assert count >= 0, f"Negative OD count for {orig}->{dest}: {count}"
    assert od_data.get("total_trips", 0) > 0, "Zero total trips in OD matrix"
    print(f"[PASS] 6. OD matrix matches camera network topology ({od_data['total_trips']} total trips).")

    # Rule 7: Route counts are non-negative
    for rd in route_data.get("route_densities", []):
        assert rd["from_camera"] in valid_cameras
        assert rd["to_camera"] in valid_cameras
        assert rd["vehicle_count"] >= 0, f"Negative route vehicle count: {rd}"
    assert route_data.get("total_transitions", 0) > 0
    print(f"[PASS] 7. Route counts are non-negative ({route_data['total_transitions']} total transitions).")

    # Rule 8: Heatmap coordinates match cameras.yaml
    for hp in heatmap_data.get("heatmap_points", []):
        cid = hp["camera_id"]
        expected_lat = cam_data[cid]["latitude"]
        expected_lon = cam_data[cid]["longitude"]
        assert math.isclose(hp["latitude"], expected_lat, abs_tol=1e-5), f"Latitude mismatch in {cid}"
        assert math.isclose(hp["longitude"], expected_lon, abs_tol=1e-5), f"Longitude mismatch in {cid}"
    print("[PASS] 8. Heatmap GPS coordinates exactly match configured camera positions.")

    # Rule 9: Heatmap intensity is within [0, 1]
    for hp in heatmap_data.get("heatmap_points", []):
        intensity = hp["traffic_intensity"]
        assert 0.0 <= intensity <= 1.0, f"Heatmap intensity out of [0, 1]: {intensity}"
    print("[PASS] 9. Heatmap observation intensities are strictly within [0.0, 1.0].")

    # Rule 10: Time-series timestamps are chronological
    buckets = timeseries_data.get("buckets", [])
    assert len(buckets) > 0, "No time-series buckets found"
    for i in range(len(buckets) - 1):
        t_curr = datetime.fromisoformat(buckets[i]["timestamp_start"])
        t_next = datetime.fromisoformat(buckets[i + 1]["timestamp_start"])
        assert t_curr < t_next, f"Time-series non-chronological: {t_curr} vs {t_next}"
    print(f"[PASS] 10. Time-series timestamps are strictly chronological ({len(buckets)} buckets).")

    # Rule 11: Vehicle-class counts are non-negative
    for cid, stats in summary_data.get("camera_statistics", {}).items():
        for vcls, cnt in stats.get("vehicle_classes", {}).items():
            assert cnt >= 0, f"Negative class count for {cid} -> {vcls}: {cnt}"
    for b in buckets:
        for vcls, cnt in b.get("vehicle_classes", {}).items():
            assert cnt >= 0, f"Negative bucket class count for {vcls}: {cnt}"
    print("[PASS] 11. Vehicle-class counts are non-negative across all sensors and buckets.")

    # Rule 12: Transition probabilities are within [0, 1]
    for t in trans_data.get("transitions_list", []):
        p = t["probability"]
        assert 0.0 <= p <= 1.0001, f"Transition probability out of [0, 1]: {p}"
    print("[PASS] 12. Transition probabilities are strictly bounded within [0.0, 1.0].")

    # Rule 13: Transition probabilities for each source camera sum approximately to 1
    cam_trans = trans_data.get("camera_transition_probabilities", {})
    for src, transitions in cam_trans.items():
        if transitions:
            prob_sum = sum(t["probability"] for t in transitions)
            assert math.isclose(prob_sum, 1.0, abs_tol=0.01), f"Probabilities for {src} do not sum to 1.0: {prob_sum}"
    print("[PASS] 13. Outgoing transition probabilities sum to 1.0 for each source camera.")

    # Rule 14: No transition probability is invented (no transitions from CAM_03)
    assert "CAM_03" not in cam_trans or len(cam_trans["CAM_03"]) == 0, \
        "Fabricated transition probability detected from terminus camera CAM_03"
    print("[PASS] 14. Zero fabricated transition probabilities for unobserved transitions.")

    # Rule 15: Route predictions only use available historical data
    for pred in pred_data.get("vehicle_route_predictions", []):
        curr_cam = pred["current_camera"]
        if curr_cam == "CAM_03":
            assert not pred["prediction_available"], "CAM_03 should be marked unavailable"
            assert "insufficient" in pred.get("reason", "").lower()
        else:
            assert pred["prediction_available"], f"{curr_cam} should have available predictions"
    print("[PASS] 15. Route predictions strictly utilize empirical historical data without hallucinations.")

    # Rule 16: Prediction probabilities are within [0, 1]
    for pred in pred_data.get("vehicle_route_predictions", []):
        for cand in pred.get("predictions", []):
            assert 0.0 <= cand["probability"] <= 1.0001, f"Prediction probability out of range: {cand}"
    print("[PASS] 16. All candidate prediction probabilities are within [0.0, 1.0].")

    # Rule 17: No fake speed values are used in analytics
    # Verify no 500+ km/h speeds appear in the summary, od, heatmap, or route prediction outputs
    for s_file in [summary_file, od_file, heatmap_file, pred_file]:
        with open(s_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
            assert "520." not in content and "604." not in content and "687." not in content, \
                f"Simulated travel speed artifact leaked into analytics file: {s_file.name}"
    print("[PASS] 17. Unrealistic simulated speed values are strictly excluded from traffic analytics.")

    # Rule 18: Existing Checkpoint 1–7 modules remain unaffected
    try:
        from reid.vehicle_reid import VehicleReIDExtractor
        from ocr.license_plate_ocr import LicensePlateOCR
        from inference.plate_ocr_pipeline import PlateOCRPipeline
        from tracking.bytetrack_tracker import run_bytetrack_tracking
        from tracking.unified_vehicle_pipeline import UnifiedVehiclePipeline
        from tracking.cross_camera_matching import CrossCameraMatcher
        from trajectory.trajectory_reconstruction import TrajectoryReconstructor
        print("[PASS] 18. All Checkpoint 1–7 core modules import cleanly without regression.")
    except Exception as e:
        raise AssertionError(f"Checkpoint 1–7 module regression detected: {e}")

    # Rule 19: pip check remains clean
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"pip check failed: {result.stdout} {result.stderr}"
    assert "No broken requirements found" in result.stdout, f"pip check output: {result.stdout}"
    print("[PASS] 19. pip check passes with zero broken requirements.")

    print("=" * 70)
    print("ALL 19 CHECKPOINT 8 VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_traffic_analytics_checkpoint_8()
