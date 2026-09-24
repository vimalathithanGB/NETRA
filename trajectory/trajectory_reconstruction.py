"""
NETRA — Spatial-Temporal Vehicle Trajectory Reconstruction & GIS Camera Mapping
=============================================================================
Module: trajectory/trajectory_reconstruction.py

Description:
    Processes resolved global vehicle entities and camera observations to reconstruct
    chronologically ordered spatial-temporal trajectories across the camera network.
    Calculates geographic road distances via the Haversine formula, inter-camera
    travel times, transit speeds, road speed-limit feasibility, and compass movement
    bearings. Generates interactive Leaflet map visualizations and structured telemetry.

Scope Boundaries:
    - Strictly trajectory & GIS reconstruction (no databases, no VLM, no legal speeding claims).
    - Preserves exact frame indexes alongside synthetic timestamp anchors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [NETRA.Trajectory] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NETRA.Trajectory")


# =============================================================================
# CONSTANTS & DEFAULTS
# =============================================================================

EARTH_RADIUS_METERS = 6371000.0  # WGS-84 mean spherical radius
DEFAULT_VIDEO_START_TIME = "2026-01-01T09:30:00"
DEFAULT_CAMERAS_CONFIG = "configs/cameras.yaml"
DEFAULT_ENTITIES_JSON = "runs/cross_camera/global_vehicle_entities.json"
DEFAULT_OBSERVATIONS_JSON = "runs/cross_camera/cross_camera_observations.json"
DEFAULT_MATCHES_JSON = "runs/cross_camera/cross_camera_matches.json"
DEFAULT_VIDEO_SOURCE = "data/videos/test_2.mp4"
DEFAULT_OUTPUT_DIR = "runs/trajectory"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CameraNode:
    """Represents a physical/simulated surveillance camera node."""
    camera_id: str
    latitude: float
    longitude: float
    road_name: str
    speed_limit_kmh: float
    description: str = ""


@dataclass
class TrajectoryObservation:
    """Observation record enriched with spatial coordinates and timestamps."""
    global_vehicle_id: str
    camera_id: str
    track_id: int
    frame_start: int
    frame_end: int
    timestamp_start: str
    timestamp_end: str
    latitude: float
    longitude: float
    vehicle_class: str
    plate_text: Optional[str] = None
    plate_status: str = "unknown"
    plate_confidence: Optional[float] = None
    last_bbox: Optional[List[int]] = None
    detector_confidence: Optional[float] = None
    observation_count: int = 1
    valid_observation_count: int = 0
    best_ocr_confidence: Optional[float] = None


@dataclass
class TrajectorySegment:
    """Represents transit between two consecutive camera observations."""
    global_vehicle_id: str
    from_camera: str
    to_camera: str
    from_timestamp: str
    to_timestamp: str
    distance_meters: float
    cumulative_distance_meters: float
    travel_time_seconds: float
    average_speed_kmh: Optional[float]
    speed_limit_kmh: float
    minimum_feasible_time_seconds: float
    travel_time_feasible: bool
    bearing_degrees: float
    direction: str
    match_score: Optional[float] = None
    reid_similarity: Optional[float] = None
    available_evidence: Optional[List[str]] = None


@dataclass
class VehicleTrajectory:
    """Full spatial-temporal journey record for a global vehicle entity."""
    global_vehicle_id: str
    vehicle_class: str
    plate_text: Optional[str]
    camera_sequence: List[str]
    observations: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    total_distance_meters: float
    total_travel_time_seconds: float
    average_speed_kmh: Optional[float]


# =============================================================================
# GEODESIC & SPATIAL CALCULATIONS
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two geographic coordinates using the
    spherical Haversine formula on Earth (radius = 6,371,000 m).

    Returns distance in meters.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    # Numerical safeguard for domain of atan2/sqrt
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, str]:
    """
    Computes initial movement bearing (forward azimuth) from point 1 to point 2.

    Returns:
        bearing_degrees: Azimuth in degrees [0, 360) where 0=N, 90=E, 180=S, 270=W.
        direction: 8-point compass cardinal string (N, NE, E, SE, S, SW, W, NW).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)

    y = math.sin(dlam) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    )

    initial_bearing = math.atan2(y, x)
    deg = (math.degrees(initial_bearing) + 360.0) % 360.0

    cardinals = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    cardinal_idx = int((deg + 22.5) // 45.0) % 8
    direction = cardinals[cardinal_idx]

    return round(deg, 2), direction


def get_actual_video_fps(video_path: str, fallback_fps: float = 29.97003) -> float:
    """
    Retrieves the actual video FPS directly from video container metadata.
    Avoids hardcoding 30 FPS when metadata indicates 29.97.
    """
    p = Path(video_path).resolve()
    if p.exists():
        cap = cv2.VideoCapture(str(p))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if 1.0 <= fps <= 120.0:
                logger.info(f"Video '{p.name}' metadata FPS: {fps:.6f}")
                return float(fps)
    logger.warning(f"Using fallback video FPS: {fallback_fps}")
    return fallback_fps


def frame_to_iso_timestamp(frame_idx: int, fps: float, anchor_time: datetime) -> str:
    """Converts a zero-indexed frame count into an ISO 8601 formatted timestamp string."""
    dt = anchor_time + timedelta(seconds=float(frame_idx) / fps)
    return dt.isoformat()


# =============================================================================
# RECONSTRUCTION ENGINE
# =============================================================================

class TrajectoryReconstructor:
    """
    Spatial-temporal trajectory reconstruction and GIS camera mapping engine.
    """

    def __init__(
        self,
        cameras_config_path: str = DEFAULT_CAMERAS_CONFIG,
        video_start_time: str = DEFAULT_VIDEO_START_TIME,
        fps: float = 29.97003,
    ):
        self.cameras = self.load_camera_config(cameras_config_path)
        self.video_start_time = datetime.fromisoformat(video_start_time)
        self.fps = fps

    @staticmethod
    def load_camera_config(config_path: str) -> Dict[str, CameraNode]:
        """Loads and validates camera network GIS coordinates and road speed limits."""
        path = Path(config_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Camera configuration not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cam_dict = data.get("cameras", {})
        if not cam_dict:
            raise ValueError(f"No cameras defined in configuration file: {path}")

        cameras: Dict[str, CameraNode] = {}
        for cam_id, cfg in cam_dict.items():
            cameras[cam_id] = CameraNode(
                camera_id=cam_id,
                latitude=float(cfg["latitude"]),
                longitude=float(cfg["longitude"]),
                road_name=str(cfg.get("road_name", "UNKNOWN")),
                speed_limit_kmh=float(cfg.get("speed_limit_kmh", 40.0)),
                description=str(cfg.get("description", "")),
            )
        logger.info(f"Loaded {len(cameras)} camera nodes from: {path.name}")
        return cameras

    def reconstruct_trajectories(
        self,
        entities_path: str,
        observations_path: str,
        matches_path: Optional[str] = "runs/cross_camera/cross_camera_matches.json",
    ) -> Tuple[List[VehicleTrajectory], List[TrajectorySegment]]:
        """
        Reconstructs spatial-temporal trajectories for all global vehicle entities.
        """
        ent_file = Path(entities_path).resolve()
        obs_file = Path(observations_path).resolve()

        if not ent_file.exists():
            raise FileNotFoundError(f"Global entities file not found: {ent_file}")
        if not obs_file.exists():
            raise FileNotFoundError(f"Observations file not found: {obs_file}")

        with open(ent_file, "r", encoding="utf-8") as f:
            entities_data = json.load(f)
        with open(obs_file, "r", encoding="utf-8") as f:
            observations_data = json.load(f)

        entities_list = entities_data.get("global_vehicles", [])
        obs_list = observations_data.get("camera_observations", [])

        # Build lookup table for observations: (camera_id, track_id) -> observation dict
        obs_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {
            (o["camera_id"], int(o["track_id"])): o for o in obs_list
        }

        # Build lookup table for cross-camera pairwise match explanations
        matches_lookup: Dict[Tuple[str, int, str, int], Dict[str, Any]] = {}
        if matches_path and Path(matches_path).exists():
            with open(matches_path, "r", encoding="utf-8") as f:
                matches_data = json.load(f).get("matches", [])
                for m in matches_data:
                    matches_lookup[(m["camera_a"], int(m["track_a"]), m["camera_b"], int(m["track_b"]))] = m

        all_trajectories: List[VehicleTrajectory] = []
        all_segments: List[TrajectorySegment] = []

        for entity in entities_list:
            gid = entity["global_vehicle_id"]
            vclass = entity["vehicle_class"]
            vplate = entity.get("plate_text")

            # 1. Retrieve all observation records belonging to this global vehicle
            entity_obs_records: List[TrajectoryObservation] = []
            for obs_ref in entity.get("observations", []):
                cam_id = obs_ref["camera_id"]
                tid = int(obs_ref["track_id"])

                full_obs = obs_lookup.get((cam_id, tid))
                if full_obs is None:
                    logger.warning(f"Entity {gid}: Observation ({cam_id}, {tid}) not found in lookup!")
                    continue

                if cam_id not in self.cameras:
                    raise KeyError(f"Camera ID '{cam_id}' not found in camera configuration!")

                cam_node = self.cameras[cam_id]
                f_start = int(full_obs["first_seen_frame"])
                f_end = int(full_obs["last_seen_frame"])

                t_start_iso = frame_to_iso_timestamp(f_start, self.fps, self.video_start_time)
                t_end_iso = frame_to_iso_timestamp(f_end, self.fps, self.video_start_time)

                obs_record = TrajectoryObservation(
                    global_vehicle_id=gid,
                    camera_id=cam_id,
                    track_id=tid,
                    frame_start=f_start,
                    frame_end=f_end,
                    timestamp_start=t_start_iso,
                    timestamp_end=t_end_iso,
                    latitude=cam_node.latitude,
                    longitude=cam_node.longitude,
                    vehicle_class=full_obs.get("vehicle_class", vclass),
                    plate_text=full_obs.get("plate_text") or vplate,
                    plate_status=full_obs.get("plate_status", "unknown"),
                    plate_confidence=float(full_obs["plate_confidence"]) if full_obs.get("plate_confidence") is not None else None,
                    last_bbox=full_obs.get("last_bbox"),
                    detector_confidence=float(full_obs["detector_confidence"]) if full_obs.get("detector_confidence") is not None else None,
                    observation_count=int(full_obs.get("observation_count", 1)),
                    valid_observation_count=int(full_obs.get("valid_observation_count", 0)),
                    best_ocr_confidence=float(full_obs["best_ocr_confidence"]) if full_obs.get("best_ocr_confidence") is not None else None,
                )
                entity_obs_records.append(obs_record)

            # 2. Sort observations chronologically by frame_start
            entity_obs_records.sort(key=lambda o: (o.frame_start, o.camera_id))

            # 3. Extract camera sequence
            cam_sequence = [o.camera_id for o in entity_obs_records]

            # 4. Compute trajectory segments for multi-camera observations
            entity_segments: List[TrajectorySegment] = []
            total_dist_m = 0.0

            if len(entity_obs_records) >= 2:
                cum_dist_m = 0.0
                for i in range(len(entity_obs_records) - 1):
                    obs_a = entity_obs_records[i]
                    obs_b = entity_obs_records[i + 1]

                    # Road distance via Haversine
                    d_m = haversine_distance(
                        obs_a.latitude, obs_a.longitude,
                        obs_b.latitude, obs_b.longitude,
                    )
                    cum_dist_m += d_m

                    # Travel time between camera observation arrival timestamps
                    dt_a = datetime.fromisoformat(obs_a.timestamp_start)
                    dt_b = datetime.fromisoformat(obs_b.timestamp_start)
                    travel_time_sec = (dt_b - dt_a).total_seconds()

                    # Speed and feasibility calculation
                    cam_b_node = self.cameras[obs_b.camera_id]
                    speed_limit_kmh = cam_b_node.speed_limit_kmh
                    speed_limit_mps = speed_limit_kmh / 3.6

                    min_feasible_sec = d_m / speed_limit_mps if speed_limit_mps > 0 else 0.0

                    if travel_time_sec > 0:
                        speed_mps = d_m / travel_time_sec
                        speed_kmh = speed_mps * 3.6
                        travel_feasible = travel_time_sec >= min_feasible_sec
                    else:
                        speed_kmh = None
                        travel_feasible = False

                    bearing_deg, direction_cardinal = calculate_bearing(
                        obs_a.latitude, obs_a.longitude,
                        obs_b.latitude, obs_b.longitude,
                    )

                    m_info = matches_lookup.get((obs_a.camera_id, obs_a.track_id, obs_b.camera_id, obs_b.track_id))
                    seg_match_score = float(m_info["match_score"]) if m_info and "match_score" in m_info else None
                    seg_reid_sim = float(m_info["reid_similarity"]) if m_info and m_info.get("reid_similarity") is not None else None
                    seg_evidence = list(m_info["available_evidence"]) if m_info and m_info.get("available_evidence") else None

                    seg = TrajectorySegment(
                        global_vehicle_id=gid,
                        from_camera=obs_a.camera_id,
                        to_camera=obs_b.camera_id,
                        from_timestamp=obs_a.timestamp_start,
                        to_timestamp=obs_b.timestamp_start,
                        distance_meters=round(d_m, 2),
                        cumulative_distance_meters=round(cum_dist_m, 2),
                        travel_time_seconds=round(travel_time_sec, 3),
                        average_speed_kmh=round(speed_kmh, 2) if speed_kmh is not None else None,
                        speed_limit_kmh=round(speed_limit_kmh, 1),
                        minimum_feasible_time_seconds=round(min_feasible_sec, 3),
                        travel_time_feasible=travel_feasible,
                        bearing_degrees=bearing_deg,
                        direction=direction_cardinal,
                        match_score=seg_match_score,
                        reid_similarity=seg_reid_sim,
                        available_evidence=seg_evidence,
                    )
                    entity_segments.append(seg)
                    all_segments.append(seg)

                total_dist_m = cum_dist_m

                # Total travel time from the first arrival to the last arrival (do not sum overlapping)
                first_dt = datetime.fromisoformat(entity_obs_records[0].timestamp_start)
                last_dt = datetime.fromisoformat(entity_obs_records[-1].timestamp_start)
                total_time_sec = max(0.0, (last_dt - first_dt).total_seconds())

                avg_speed_kmh = (
                    round((total_dist_m / total_time_sec) * 3.6, 2)
                    if total_time_sec > 0 else None
                )
            else:
                # Single camera observation: distance is 0
                total_dist_m = 0.0
                if entity_obs_records:
                    dt_s = datetime.fromisoformat(entity_obs_records[0].timestamp_start)
                    dt_e = datetime.fromisoformat(entity_obs_records[0].timestamp_end)
                    total_time_sec = round(max(0.0, (dt_e - dt_s).total_seconds()), 3)
                else:
                    total_time_sec = 0.0
                avg_speed_kmh = 0.0

            traj = VehicleTrajectory(
                global_vehicle_id=gid,
                vehicle_class=vclass,
                plate_text=vplate,
                camera_sequence=cam_sequence,
                observations=[asdict(o) for o in entity_obs_records],
                segments=[asdict(s) for s in entity_segments],
                total_distance_meters=round(total_dist_m, 2),
                total_travel_time_seconds=round(total_time_sec, 3),
                average_speed_kmh=avg_speed_kmh,
            )
            all_trajectories.append(traj)

        return all_trajectories, all_segments


# =============================================================================
# LEAFLET MAP VISUALIZATION GENERATOR
# =============================================================================

def generate_trajectory_map_html(
    cameras: Dict[str, CameraNode],
    trajectories: List[VehicleTrajectory],
    output_html_path: str,
) -> None:
    """
    Generates a standalone, interactive Leaflet HTML map visualizing camera nodes,
    road corridors, and reconstructed multi-camera vehicle trajectories.
    """
    out_file = Path(output_html_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Filter trajectories with at least 2 camera observations
    multi_cam_trajs = [t for t in trajectories if len(t.camera_sequence) >= 2]

    # Palette of distinctive colors for vehicle paths
    palette = [
        "#00E5FF", "#FF3D00", "#76FF03", "#FFD600", "#D500F9",
        "#00E676", "#FF1744", "#2979FF", "#FF9100", "#651FFF",
        "#1DE9B6", "#F50057", "#00B0FF", "#C6FF00", "#FF6D00"
    ]

    # Build GeoJSON or JS structure for camera markers
    camera_js_data = []
    for cam_id, cam in cameras.items():
        camera_js_data.append({
            "camera_id": cam.camera_id,
            "latitude": cam.latitude,
            "longitude": cam.longitude,
            "road_name": cam.road_name,
            "speed_limit_kmh": cam.speed_limit_kmh,
            "description": cam.description,
        })

    # Build trajectory paths for JS
    trajectory_js_data = []
    for idx, t in enumerate(multi_cam_trajs):
        coords = [[obs["latitude"], obs["longitude"]] for obs in t.observations]
        color = palette[idx % len(palette)]
        trajectory_js_data.append({
            "global_vehicle_id": t.global_vehicle_id,
            "vehicle_class": t.vehicle_class,
            "plate_text": t.plate_text or "N/A",
            "camera_sequence": " → ".join(t.camera_sequence),
            "coordinates": coords,
            "color": color,
            "total_distance_m": t.total_distance_meters,
            "total_time_s": t.total_travel_time_seconds,
            "average_speed_kmh": t.average_speed_kmh if t.average_speed_kmh is not None else 0.0,
            "num_cameras": len(t.camera_sequence),
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NETRA — Multi-Camera Trajectory & GIS Network Map</title>
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body, html {{
            height: 100%;
            width: 100%;
            background-color: #0b0f19;
            color: #e2e8f0;
            overflow: hidden;
        }}
        #app-header {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: rgba(15, 23, 42, 0.92);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            z-index: 1000;
        }}
        .brand-title {{
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: #38bdf8;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .brand-badge {{
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid #38bdf8;
            color: #38bdf8;
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .disclaimer-banner {{
            font-size: 0.76rem;
            color: #fbbf24;
            background: rgba(251, 191, 36, 0.12);
            border: 1px solid rgba(251, 191, 36, 0.3);
            padding: 4px 12px;
            border-radius: 4px;
        }}
        #map {{
            height: 100%;
            width: 100%;
            padding-top: 60px;
            z-index: 1;
        }}
        /* Info Sidebar Panel */
        #sidebar {{
            position: absolute;
            top: 75px;
            right: 20px;
            width: 340px;
            max-height: calc(100% - 95px);
            background: rgba(15, 23, 42, 0.94);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .sidebar-header {{
            padding: 14px 18px;
            background: rgba(30, 41, 59, 0.8);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.9rem;
            font-weight: 600;
            color: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .sidebar-list {{
            padding: 12px;
            overflow-y: auto;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .traj-card {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 10px 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .traj-card:hover {{
            border-color: #38bdf8;
            background: rgba(30, 41, 59, 0.9);
            transform: translateY(-2px);
        }}
        .traj-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .traj-gid {{
            font-weight: 700;
            color: #38bdf8;
            font-size: 0.85rem;
        }}
        .traj-class {{
            font-size: 0.72rem;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(148, 163, 184, 0.15);
            color: #cbd5e1;
            text-transform: uppercase;
        }}
        .traj-detail {{
            font-size: 0.75rem;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
            margin-top: 3px;
        }}
        .custom-popup .leaflet-popup-content-wrapper {{
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(12px);
            color: #e2e8f0;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 10px;
        }}
        .custom-popup .leaflet-popup-tip {{
            background: rgba(15, 23, 42, 0.95);
        }}
    </style>
</head>
<body>
    <header id="app-header">
        <div class="brand-title">
            <span>NETRA AI ENGINE</span>
            <span class="brand-badge">Checkpoint 7 — Trajectory & GIS</span>
        </div>
        <div class="disclaimer-banner">
            ⚠️ SIMULATION MODE: Camera GPS and video timestamps are synthetic demonstration values.
        </div>
    </header>

    <div id="sidebar">
        <div class="sidebar-header">
            <span>Reconstructed Trajectories ({len(multi_cam_trajs)})</span>
            <span style="font-size: 0.75rem; color: #38bdf8;">3 Cameras Active</span>
        </div>
        <div class="sidebar-list" id="traj-list"></div>
    </div>

    <div id="map"></div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
            integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script>
        const cameraData = {json.dumps(camera_js_data, indent=2)};
        const trajectoryData = {json.dumps(trajectory_js_data, indent=2)};

        // Initialize Leaflet Map centered around simulated camera cluster
        const map = L.map('map', {{
            center: [11.0195, 76.9584],
            zoom: 16,
            zoomControl: true
        }});

        // Dark Matter / CartoDB Dark Base Layer
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; <a href="https://carto.com/">CARTO</a> | NETRA GIS Simulation',
            subdomains: 'abcd',
            maxZoom: 19
        }}).addTo(map);

        // Render Camera Nodes with pulsing circle markers
        cameraData.forEach(cam => {{
            const marker = L.circleMarker([cam.latitude, cam.longitude], {{
                radius: 10,
                fillColor: '#38bdf8',
                color: '#ffffff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.85
            }}).addTo(map);

            const popupContent = `
                <div style="font-size: 0.85rem; padding: 4px;">
                    <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; margin-bottom: 4px;">
                        🎥 ${{cam.camera_id}}
                    </div>
                    <div style="color: #cbd5e1; margin-bottom: 2px;"><b>Road:</b> ${{cam.road_name}}</div>
                    <div style="color: #cbd5e1; margin-bottom: 2px;"><b>Speed Limit:</b> ${{cam.speed_limit_kmh}} km/h</div>
                    <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                        GPS: ${{cam.latitude.toFixed(4)}}, ${{cam.longitude.toFixed(4)}}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 2px;">
                        ${{cam.description}}
                    </div>
                </div>
            `;
            marker.bindPopup(popupContent, {{ className: 'custom-popup' }});
        }});

        // Render Reconstructed Vehicle Trajectories as Polylines
        const polylines = {{}};
        trajectoryData.forEach(t => {{
            const polyline = L.polyline(t.coordinates, {{
                color: t.color,
                weight: 4,
                opacity: 0.8,
                smoothFactor: 1
            }}).addTo(map);

            const trajPopup = `
                <div style="font-size: 0.85rem; padding: 4px;">
                    <div style="font-weight: 700; color: ${{t.color}}; font-size: 0.95rem; margin-bottom: 4px;">
                        🚗 ${{t.global_vehicle_id}} (${{t.vehicle_class.toUpperCase()}})
                    </div>
                    <div style="color: #cbd5e1; margin-bottom: 2px;"><b>Route:</b> ${{t.camera_sequence}}</div>
                    <div style="color: #cbd5e1; margin-bottom: 2px;"><b>Distance:</b> ${{t.total_distance_m}} meters</div>
                    <div style="color: #cbd5e1; margin-bottom: 2px;"><b>Transit Time:</b> ${{t.total_time_s}} s</div>
                    <div style="color: #cbd5e1; margin-bottom: 2px;"><b>Avg Speed:</b> ${{t.average_speed_kmh}} km/h</div>
                    <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">Plate: ${{t.plate_text}}</div>
                </div>
            `;
            polyline.bindPopup(trajPopup, {{ className: 'custom-popup' }});
            polylines[t.global_vehicle_id] = polyline;
        }});

        // Populate Sidebar Trajectory List
        const listContainer = document.getElementById('traj-list');
        trajectoryData.forEach(t => {{
            const card = document.createElement('div');
            card.className = 'traj-card';
            card.innerHTML = `
                <div class="traj-card-header">
                    <span class="traj-gid" style="color: ${{t.color}};">${{t.global_vehicle_id}}</span>
                    <span class="traj-class">${{t.vehicle_class}}</span>
                </div>
                <div class="traj-detail">
                    <span>Route:</span>
                    <span style="font-weight: 600; color: #f1f5f9;">${{t.camera_sequence}}</span>
                </div>
                <div class="traj-detail">
                    <span>Distance: ${{t.total_distance_m}}m</span>
                    <span>Speed: ${{t.average_speed_kmh}} km/h</span>
                </div>
            `;
            card.addEventListener('click', () => {{
                if (polylines[t.global_vehicle_id]) {{
                    map.fitBounds(polylines[t.global_vehicle_id].getBounds(), {{ padding: [60, 60] }});
                    polylines[t.global_vehicle_id].openPopup();
                }}
            }});
            listContainer.appendChild(card);
        }});
    </script>
</body>
</html>
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Generated interactive Leaflet map: {out_file}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="NETRA Checkpoint 7: Spatial-Temporal Trajectory Reconstruction & GIS Mapping"
    )
    parser.add_argument(
        "--cameras-config",
        type=str,
        default=DEFAULT_CAMERAS_CONFIG,
        help="Path to cameras YAML configuration file.",
    )
    parser.add_argument(
        "--entities-json",
        type=str,
        default=DEFAULT_ENTITIES_JSON,
        help="Path to global_vehicle_entities.json from Checkpoint 6.",
    )
    parser.add_argument(
        "--observations-json",
        type=str,
        default=DEFAULT_OBSERVATIONS_JSON,
        help="Path to cross_camera_observations.json from Checkpoint 6.",
    )
    parser.add_argument(
        "--video-source",
        type=str,
        default=DEFAULT_VIDEO_SOURCE,
        help="Path to source video file for actual FPS metadata extraction.",
    )
    parser.add_argument(
        "--video-start-time",
        type=str,
        default=DEFAULT_VIDEO_START_TIME,
        help="Synthetic ISO 8601 anchor timestamp for frame conversion.",
    )
    parser.add_argument(
        "--matches-json",
        type=str,
        default=DEFAULT_MATCHES_JSON,
        help="Path to cross_camera_matches.json for match evidence enrichment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save reconstructed trajectories and maps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine actual video FPS
    actual_fps = get_actual_video_fps(args.video_source)

    # 2. Instantiate reconstructor
    reconstructor = TrajectoryReconstructor(
        cameras_config_path=args.cameras_config,
        video_start_time=args.video_start_time,
        fps=actual_fps,
    )

    # 3. Execute trajectory and segment reconstruction
    trajectories, segments = reconstructor.reconstruct_trajectories(
        entities_path=args.entities_json,
        observations_path=args.observations_json,
        matches_path=args.matches_json,
    )

    # 4. Save JSON outputs
    traj_out_path = out_dir / "vehicle_trajectories.json"
    with open(traj_out_path, "w", encoding="utf-8") as f:
        json.dump({"vehicle_trajectories": [asdict(t) for t in trajectories]}, f, indent=2)
    logger.info(f"Saved vehicle trajectories JSON: {traj_out_path}")

    seg_out_path = out_dir / "trajectory_segments.json"
    with open(seg_out_path, "w", encoding="utf-8") as f:
        json.dump({"trajectory_segments": [asdict(s) for s in segments]}, f, indent=2)
    logger.info(f"Saved trajectory segments JSON: {seg_out_path}")

    # 5. Generate interactive HTML Leaflet Map
    map_out_path = out_dir / "trajectory_map.html"
    generate_trajectory_map_html(
        cameras=reconstructor.cameras,
        trajectories=trajectories,
        output_html_path=str(map_out_path),
    )

    # 6. Print console summary
    multi_cam = [t for t in trajectories if len(t.camera_sequence) >= 2]
    print("\n" + "=" * 70)
    print("NETRA TRAJECTORY RECONSTRUCTION EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Global Vehicles:        {len(trajectories)}")
    print(f"Multi-Camera Trajectories:    {len(multi_cam)}")
    print(f"Single-Camera Entities:       {len(trajectories) - len(multi_cam)}")
    print(f"Total Trajectory Segments:    {len(segments)}")
    if segments:
        avg_dist = sum(s.distance_meters for s in segments) / len(segments)
        print(f"Mean Segment Distance:        {avg_dist:.2f} meters")
    print(f"Saved Trajectories:           {traj_out_path}")
    print(f"Saved Segments:               {seg_out_path}")
    print(f"Saved Interactive Map:        {map_out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
