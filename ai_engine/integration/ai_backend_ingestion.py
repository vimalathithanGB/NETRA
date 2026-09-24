"""NETRA AI Ingestion Runner.

Reads confirmed AI Engine pipeline outputs from runs/ directory and
ingests them into the NETRA FastAPI backend via HTTP REST APIs.
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

from ai_engine.integration.backend_client import NETRABackendClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("NETRA-AIIngestRunner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_json(filepath: Path) -> Any:
    """Safely load JSON from file."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_track(t: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize observation track to match VehicleTrackInput schema."""
    return {
        "track_id": int(t["track_id"]),
        "vehicle_class": str(t.get("vehicle_class", "unknown")),
        "plate_text": t.get("plate_text"),
        "plate_status": t.get("plate_status") or "unknown",
        "plate_confidence": float(t["plate_confidence"]) if t.get("plate_confidence") is not None else None,
        "observation_count": int(t.get("observation_count", 0)),
        "valid_observation_count": int(t.get("valid_observation_count", 0)),
        "best_ocr_confidence": float(t["best_ocr_confidence"]) if t.get("best_ocr_confidence") is not None else None,
        "first_seen_frame": int(t["first_seen_frame"]) if t.get("first_seen_frame") is not None else None,
        "last_seen_frame": int(t["last_seen_frame"]) if t.get("last_seen_frame") is not None else None,
    }


def sanitize_match(m: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize cross-camera match to match CrossCameraMatchInput schema."""
    return {
        "camera_a": str(m["camera_a"]),
        "track_a": int(m["track_a"]),
        "camera_b": str(m["camera_b"]),
        "track_b": int(m["track_b"]),
        "reid_similarity": float(m["reid_similarity"]) if m.get("reid_similarity") is not None else None,
        "plate_match": bool(m["plate_match"]) if m.get("plate_match") is not None else None,
        "class_match": float(m["class_match"]) if m.get("class_match") is not None else None,
        "available_evidence": m.get("available_evidence") or [],
        "match_score": float(m["match_score"]),
        "threshold": float(m["threshold"]),
        "matched": bool(m["matched"]),
    }


def sanitize_global_vehicle(gv: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize global vehicle entity to match GlobalVehicleInput schema."""
    observations = []
    for obs in gv.get("observations", []):
        observations.append({
            "camera_id": str(obs["camera_id"]),
            "track_id": int(obs["track_id"]),
        })

    return {
        "global_vehicle_id": str(gv["global_vehicle_id"]),
        "vehicle_class": str(gv.get("vehicle_class", "unknown")),
        "observations": observations,
        "plate_text": gv.get("plate_text"),
        "match_confidence": float(gv["match_confidence"]) if gv.get("match_confidence") is not None else None,
    }


def sanitize_trajectory(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize trajectory to match TrajectoryInput schema."""
    segments = []
    for s in traj.get("segments", []):
        segments.append({
            "global_vehicle_id": s.get("global_vehicle_id"),
            "from_camera": str(s["from_camera"]),
            "to_camera": str(s["to_camera"]),
            "from_timestamp": s.get("from_timestamp"),
            "to_timestamp": s.get("to_timestamp"),
            "distance_meters": float(s.get("distance_meters", 0.0)),
            "cumulative_distance_meters": float(s.get("cumulative_distance_meters", 0.0)),
            "travel_time_seconds": float(s.get("travel_time_seconds", 0.0)),
            "average_speed_kmh": float(s.get("average_speed_kmh", 0.0)),
            "speed_limit_kmh": float(s.get("speed_limit_kmh", 0.0)),
            "minimum_feasible_time_seconds": float(s.get("minimum_feasible_time_seconds", 0.0)),
            "travel_time_feasible": bool(s.get("travel_time_feasible", True)),
            "bearing_degrees": float(s.get("bearing_degrees", 0.0)),
            "direction": s.get("direction"),
        })

    return {
        "global_vehicle_id": str(traj["global_vehicle_id"]),
        "vehicle_class": str(traj.get("vehicle_class", "unknown")),
        "plate_text": traj.get("plate_text"),
        "camera_sequence": [str(c) for c in traj.get("camera_sequence", [])],
        "total_distance_meters": float(traj.get("total_distance_meters", 0.0)),
        "total_travel_time_seconds": float(traj.get("total_travel_time_seconds", 0.0)),
        "average_speed_kmh": float(traj.get("average_speed_kmh", 0.0)),
        "segments": segments,
    }


def sanitize_route_prediction(rp: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize route prediction to match RoutePredictionInput schema."""
    predictions = []
    for p in rp.get("predictions", []):
        predictions.append({
            "next_camera": str(p["next_camera"]),
            "probability": float(p["probability"]),
            "observed_historical_transitions": int(p["observed_historical_transitions"]),
        })

    return {
        "global_vehicle_id": str(rp["global_vehicle_id"]),
        "current_camera": str(rp["current_camera"]),
        "vehicle_class": str(rp.get("vehicle_class", "unknown")),
        "prediction_available": bool(rp.get("prediction_available", False)),
        "predictions": predictions,
    }


def run_ai_ingestion(backend_url: str = None, runs_dir: Path = None) -> Dict[str, Any]:
    """Execute complete ingestion of real AI pipeline outputs into backend."""
    client = NETRABackendClient(backend_url=backend_url)
    runs_dir = runs_dir or (PROJECT_ROOT / "runs")

    logger.info(f"Connecting to NETRA backend at: {client.base_url}")
    if not client.check_health():
        logger.error(f"Cannot reach backend at {client.base_url}. Aborting ingestion.")
        return {"success": False, "error": "Backend health check failed"}

    summary = {
        "success": True,
        "observations_ingested": 0,
        "matches_ingested": 0,
        "global_vehicles_ingested": 0,
        "trajectories_ingested": 0,
        "predictions_ingested": 0,
        "analytics_ingested": False,
    }

    # 1. Observations from runs/cross_camera/cross_camera_observations.json
    obs_file = runs_dir / "cross_camera" / "cross_camera_observations.json"
    obs_data = load_json(obs_file)
    if obs_data and "camera_observations" in obs_data:
        # Group by camera_id
        by_cam = {}
        for item in obs_data["camera_observations"]:
            cam_id = item["camera_id"]
            if cam_id not in by_cam:
                by_cam[cam_id] = []
            by_cam[cam_id].append(sanitize_track(item))

        for cam_id, tracks in by_cam.items():
            res = client.ingest_tracks(camera_id=cam_id, tracks=tracks)
            if res.get("success"):
                cnt = res.get("counts", {}).get("tracks", len(tracks))
                summary["observations_ingested"] += cnt
                logger.info(f"Ingested {cnt} observations for camera {cam_id}")
            else:
                logger.warning(f"Failed to ingest tracks for {cam_id}: {res}")

    # 2. Cross-Camera Matches from runs/cross_camera/cross_camera_matches.json
    matches_file = runs_dir / "cross_camera" / "cross_camera_matches.json"
    matches_data = load_json(matches_file)
    if matches_data and "matches" in matches_data:
        # Ingest matches (focusing on accepted matches or all comparisons)
        raw_matches = matches_data["matches"]
        # Ingest the accepted matches to keep db clean and performant
        accepted = [sanitize_match(m) for m in raw_matches if m.get("matched")]
        if accepted:
            res = client.ingest_matches(accepted)
            if res.get("success"):
                cnt = res.get("counts", {}).get("matches", len(accepted))
                summary["matches_ingested"] = cnt
                logger.info(f"Ingested {cnt} accepted cross-camera matches")
            else:
                logger.warning(f"Failed to ingest matches: {res}")

    # 3. Global Vehicle Entities from runs/cross_camera/global_vehicle_entities.json
    gv_file = runs_dir / "cross_camera" / "global_vehicle_entities.json"
    gv_data = load_json(gv_file)
    if gv_data and "global_vehicles" in gv_data:
        sanitized_gv = [sanitize_global_vehicle(gv) for gv in gv_data["global_vehicles"]]
        res = client.ingest_global_vehicles(sanitized_gv)
        if res.get("success"):
            cnt = res.get("counts", {}).get("global_vehicles", len(sanitized_gv))
            summary["global_vehicles_ingested"] = cnt
            logger.info(f"Ingested {cnt} global vehicle entities")
        else:
            logger.warning(f"Failed to ingest global vehicles: {res}")

    # 4. Trajectories from runs/trajectory/vehicle_trajectories.json
    traj_file = runs_dir / "trajectory" / "vehicle_trajectories.json"
    traj_data = load_json(traj_file)
    if traj_data and "vehicle_trajectories" in traj_data:
        sanitized_traj = [sanitize_trajectory(t) for t in traj_data["vehicle_trajectories"]]
        res = client.ingest_trajectories(sanitized_traj)
        if res.get("success"):
            cnt = res.get("counts", {}).get("trajectories", len(sanitized_traj))
            summary["trajectories_ingested"] = cnt
            logger.info(f"Ingested {cnt} vehicle trajectories with segments")
        else:
            logger.warning(f"Failed to ingest trajectories: {res}")

    # 5. Route Predictions from runs/analytics/route_predictions.json
    pred_file = runs_dir / "analytics" / "route_predictions.json"
    pred_data = load_json(pred_file)
    if pred_data and "vehicle_route_predictions" in pred_data:
        sanitized_preds = [sanitize_route_prediction(p) for p in pred_data["vehicle_route_predictions"]]
        res = client.ingest_predictions(sanitized_preds)
        if res.get("success"):
            cnt = res.get("counts", {}).get("predictions", len(sanitized_preds))
            summary["predictions_ingested"] = cnt
            logger.info(f"Ingested {cnt} route predictions")
        else:
            logger.warning(f"Failed to ingest route predictions: {res}")

    # 6. Consolidated Traffic Analytics from runs/analytics/traffic_analytics_summary.json
    analytics_file = runs_dir / "analytics" / "traffic_analytics_summary.json"
    analytics_data = load_json(analytics_file)
    if analytics_data:
        res = client.ingest_analytics(analytics_data)
        if res.get("success"):
            summary["analytics_ingested"] = True
            logger.info("Ingested traffic analytics summary checkpoint")
        else:
            logger.warning(f"Failed to ingest analytics summary: {res}")

    logger.info(f"Ingestion complete: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest AI Pipeline JSON outputs into NETRA backend")
    parser.add_argument("--url", default=None, help="NETRA Backend URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--runs-dir", default=None, help="Path to runs/ output directory")
    args = parser.parse_args()

    runs_path = Path(args.runs_dir) if args.runs_dir else None
    run_ai_ingestion(backend_url=args.url, runs_dir=runs_path)
