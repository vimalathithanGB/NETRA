"""
NETRA — Checkpoint 6: Cross-Camera Vehicle Re-Identification & Global Identity Resolution
Module: tracking/cross_camera_matching.py
Project: SIH 2026 NETRA (Networked Engine for Traffic Recognition & Analytics)

Architecture:
    Simulated Multi-Camera Segments (CAM_01, CAM_02, CAM_03 from test_2.mp4)
         ↓
    CameraVehicleObservation Records [camera_id, local track_id, 512-D Re-ID, plate, class]
         ↓
    Multi-Modal Evidence Fusion:
         - Visual Re-ID Cosine Similarity (Weight: 0.60)
         - License Plate Deterministic Match (Weight: 0.30)
         - Vehicle Semantic Class Match (Weight: 0.10)
         - Temporal Sequencing Constraint: last_seen(CAM_A) < first_seen(CAM_B)
         - Missing Modality Dynamic Weight Normalization
         - Strict Same-Camera Matching Prohibition (No CAM_A -> CAM_A)
         - Strong Class Conflict Rejection
         ↓
    Match Decision (MATCH_THRESHOLD = 0.75)
         ↓
    Global Identity Resolution (GV_000001, GV_000002, ...)
         ↓
    Outputs:
         - runs/cross_camera/cross_camera_observations.json
         - runs/cross_camera/global_vehicle_entities.json
         - runs/cross_camera/cross_camera_matches.json
         - runs/cross_camera/cross_camera_report.md

IMPORTANT IDENTITY & DATASET NOTICE:
    - ByteTrack Track IDs are strictly LOCAL identities within a single camera stream.
    - Global Vehicle IDs (GV_xxxxxx) represent resolved multi-camera entity clusters.
    - The camera feeds in this test are SIMULATED via temporal segmentation of test_2.mp4.
    - They are NOT physically distinct cameras. This checkpoint algorithmically validates
      the matching engine and identity resolution logic before deployment on physical camera networks.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.license_plate_ocr import clean_plate_text
from tracking.unified_vehicle_pipeline import UnifiedVehiclePipeline

# Configure module logger
logger = logging.getLogger("NETRA.CrossCamera")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [NETRA.ReID] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default Simulation Segments from test_2.mp4 (228 frames total)
DEFAULT_CAMERA_SEGMENTS = {
    "CAM_01": {"start_frame": 0, "end_frame": 75},
    "CAM_02": {"start_frame": 76, "end_frame": 151},
    "CAM_03": {"start_frame": 152, "end_frame": 227},
}

# Smoke Test Segments
SMOKE_CAMERA_SEGMENTS = {
    "CAM_01": {"start_frame": 0, "end_frame": 25},
    "CAM_02": {"start_frame": 26, "end_frame": 50},
    "CAM_03": {"start_frame": 51, "end_frame": 75},
}

# Fusion Weights
WEIGHT_REID = 0.60
WEIGHT_PLATE = 0.30
WEIGHT_CLASS = 0.10

# Default Matching Decision Threshold
DEFAULT_MATCH_THRESHOLD = 0.75


# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

@dataclass
class CameraVehicleObservation:
    """Normalized observation record from a specific camera node."""
    camera_id: str
    track_id: int
    vehicle_class: str
    first_seen_frame: int
    last_seen_frame: int
    plate_text: Optional[str]
    plate_status: str  # "stable" | "tentative" | "unknown"
    plate_confidence: float
    representative_embedding: Optional[List[float]]
    last_bbox: Optional[List[int]] = None
    detector_confidence: Optional[float] = None
    observation_count: int = 1
    valid_observation_count: int = 0
    best_ocr_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchExplanation:
    """Detailed evidence explanation for a pairwise cross-camera comparison."""
    camera_a: str
    track_a: int
    camera_b: str
    track_b: int
    reid_similarity: Optional[float]
    plate_match: Optional[float]
    class_match: float
    available_evidence: List[str]
    match_score: float
    threshold: float
    matched: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalVehicleEntity:
    """Resolved global identity cluster grouping multi-camera observations."""
    global_vehicle_id: str
    vehicle_class: str
    observations: List[Dict[str, Any]]
    plate_text: Optional[str]
    match_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 2. MATCHING & SIMILARITY FUNCTIONS
# =============================================================================

def compute_reid_cosine_similarity(
    vecA: Optional[List[float]],
    vecB: Optional[List[float]],
) -> Optional[float]:
    """
    Computes cosine similarity between two 512-D L2-normalized embeddings.
    Because vectors are unit-normalized, cosine similarity equals the dot product.

    Returns:
        float in [-1.0, 1.0] if both vectors are valid and finite, else None.
    """
    if vecA is None or vecB is None:
        return None
    if len(vecA) != 512 or len(vecB) != 512:
        return None

    arrA = np.asarray(vecA, dtype=np.float32)
    arrB = np.asarray(vecB, dtype=np.float32)

    if not np.all(np.isfinite(arrA)) or not np.all(np.isfinite(arrB)):
        return None

    dot_val = float(np.dot(arrA, arrB))
    return round(max(-1.0, min(1.0, dot_val)), 4)


def compute_plate_match(
    plateA: Optional[str],
    statusA: str,
    plateB: Optional[str],
    statusB: str,
) -> Optional[float]:
    """
    Deterministically compares license plate registration strings.

    Rules:
    - If both observations possess plate text and status is 'stable' or 'tentative':
        Exact match (after standard cleaning) -> 1.0
        Different plate text                -> 0.0
    - If either plate is None, empty, or status is 'unknown':
        Evidence is marked unavailable (None).
    """
    if not plateA or not plateB:
        return None
    if statusA == "unknown" or statusB == "unknown":
        return None

    cleanA = clean_plate_text(plateA)
    cleanB = clean_plate_text(plateB)

    if not cleanA or not cleanB:
        return None

    return 1.0 if cleanA == cleanB else 0.0


def compute_class_match(classA: str, classB: str) -> float:
    """Compares vehicle semantic classes (car, truck, bus, etc.)."""
    return 1.0 if str(classA).lower() == str(classB).lower() else 0.0


def compute_fusion_score(
    reid_sim: Optional[float],
    plate_match: Optional[float],
    class_match: float,
    weight_reid: float = WEIGHT_REID,
    weight_plate: float = WEIGHT_PLATE,
    weight_class: float = WEIGHT_CLASS,
) -> Tuple[float, List[str]]:
    """
    Calculates dynamic weighted-normalized evidence fusion score:
    Only available modalities contribute to the numerator and denominator.
    Missing modalities are NEVER penalized as zero.

    Returns:
        Tuple[float, List[str]]: (normalized_score, list_of_available_modalities)
    """
    weighted_sum = 0.0
    weight_total = 0.0
    evidence_used: List[str] = []

    # 1. Visual Re-ID Evidence
    if reid_sim is not None:
        # Cosine similarity can theoretically be negative; map [-1, 1] to [0, 1] or clamp to 0
        reid_score = max(0.0, reid_sim)
        weighted_sum += weight_reid * reid_score
        weight_total += weight_reid
        evidence_used.append("reid")

    # 2. License Plate Evidence
    if plate_match is not None:
        weighted_sum += weight_plate * plate_match
        weight_total += weight_plate
        evidence_used.append("plate")

    # 3. Vehicle Semantic Class Evidence
    weighted_sum += weight_class * class_match
    weight_total += weight_class
    evidence_used.append("class")

    if weight_total <= 0.0:
        return 0.0, evidence_used

    final_score = weighted_sum / weight_total
    return round(final_score, 4), evidence_used


# =============================================================================
# 3. CROSS-CAMERA MATCHER ENGINE
# =============================================================================

class CrossCameraMatcher:
    """
    Cross-Camera Vehicle Re-Identification & Global Identity Resolution Engine.
    """

    def __init__(
        self,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        weight_reid: float = WEIGHT_REID,
        weight_plate: float = WEIGHT_PLATE,
        weight_class: float = WEIGHT_CLASS,
    ):
        self.match_threshold = float(match_threshold)
        self.weight_reid = float(weight_reid)
        self.weight_plate = float(weight_plate)
        self.weight_class = float(weight_class)

    def compare_pair(
        self,
        obs_a: CameraVehicleObservation,
        obs_b: CameraVehicleObservation,
    ) -> MatchExplanation:
        """
        Executes a pairwise comparison between two observations:
        - Strict same-camera rejection
        - Temporal sequencing check: obs_a must precede obs_b
        - Strong class conflict rejection
        """
        # 1. Same-Camera Prohibition
        if obs_a.camera_id == obs_b.camera_id:
            return MatchExplanation(
                camera_a=obs_a.camera_id,
                track_a=obs_a.track_id,
                camera_b=obs_b.camera_id,
                track_b=obs_b.track_id,
                reid_similarity=None,
                plate_match=None,
                class_match=1.0 if obs_a.vehicle_class == obs_b.vehicle_class else 0.0,
                available_evidence=[],
                match_score=0.0,
                threshold=self.match_threshold,
                matched=False,
            )

        # 2. Compute Modality Evidences
        reid_sim = compute_reid_cosine_similarity(
            obs_a.representative_embedding,
            obs_b.representative_embedding,
        )

        plate_m = compute_plate_match(
            obs_a.plate_text, obs_a.plate_status,
            obs_b.plate_text, obs_b.plate_status,
        )

        class_m = compute_class_match(
            obs_a.vehicle_class,
            obs_b.vehicle_class,
        )

        # 3. Calculate Normalized Fusion Score
        score, evidence_used = compute_fusion_score(
            reid_sim=reid_sim,
            plate_match=plate_m,
            class_match=class_m,
            weight_reid=self.weight_reid,
            weight_plate=self.weight_plate,
            weight_class=self.weight_class,
        )

        # 4. False-Match Protection Rules
        matched = (score >= self.match_threshold)

        # Protection A: Reject if vehicle classes strongly conflict and plate is missing or conflicting
        if class_m == 0.0:
            if plate_m is None or plate_m == 0.0:
                matched = False

        # Protection B: Reject if plate matches conflicting string (plate_match == 0.0)
        if plate_m == 0.0:
            matched = False

        return MatchExplanation(
            camera_a=obs_a.camera_id,
            track_a=obs_a.track_id,
            camera_b=obs_b.camera_id,
            track_b=obs_b.track_id,
            reid_similarity=reid_sim,
            plate_match=plate_m,
            class_match=class_m,
            available_evidence=evidence_used,
            match_score=score,
            threshold=self.match_threshold,
            matched=matched,
        )

    def resolve_global_identities(
        self,
        observations: List[CameraVehicleObservation],
    ) -> Tuple[List[GlobalVehicleEntity], List[MatchExplanation]]:
        """
        Performs chronological global entity clustering:
        Iterates over camera observations in chronological order, matches against
        existing global entities, and assigns unique Global Vehicle IDs (GV_000001, etc.).

        Guarantees:
        - Every observation belongs to EXACTLY ONE global vehicle entity.
        - No observation is assigned to multiple global vehicles.
        - Global IDs are completely decoupled from local ByteTrack IDs.
        """
        # Sort observations chronologically by camera order and first_seen_frame
        sorted_obs = sorted(
            observations,
            key=lambda o: (o.first_seen_frame, o.camera_id, o.track_id)
        )

        # 1. Compute and store all valid cross-camera pairwise candidate comparisons
        all_explanations: List[MatchExplanation] = []
        for i, obs_a in enumerate(sorted_obs):
            for obs_b in sorted_obs[i + 1:]:
                # Disallow comparisons within the same camera
                if obs_a.camera_id == obs_b.camera_id:
                    continue
                # Enforce temporal ordering constraint: earlier camera must precede later
                if obs_a.last_seen_frame >= obs_b.first_seen_frame:
                    continue

                expl = self.compare_pair(obs_a, obs_b)
                all_explanations.append(expl)

        # 2. Chronological Global ID Assignment with Camera Exclusivity
        global_entities: List[GlobalVehicleEntity] = []
        next_global_idx = 1

        for curr_obs in sorted_obs:
            best_match_entity: Optional[GlobalVehicleEntity] = None
            best_match_score = -1.0

            # Find matching candidate entity among already created global entities
            for entity in global_entities:
                # Camera Exclusivity: An entity cannot contain multiple tracks from the same camera
                if any(o["camera_id"] == curr_obs.camera_id for o in entity.observations):
                    continue

                # Check evidence against observations in this candidate entity
                for prev_obs_dict in entity.observations:
                    prev_cam = prev_obs_dict["camera_id"]
                    prev_tid = prev_obs_dict["track_id"]

                    prev_full = next(
                        (o for o in observations if o.camera_id == prev_cam and o.track_id == prev_tid),
                        None
                    )
                    if prev_full is None or prev_full.last_seen_frame >= curr_obs.first_seen_frame:
                        continue

                    expl = self.compare_pair(prev_full, curr_obs)
                    if expl.matched and expl.match_score > best_match_score:
                        best_match_score = expl.match_score
                        best_match_entity = entity

            # Decision: Assign to matched global entity OR create a new global entity
            if best_match_entity is not None and best_match_score >= self.match_threshold:
                best_match_entity.observations.append({
                    "camera_id": curr_obs.camera_id,
                    "track_id": curr_obs.track_id,
                })
                # Set match confidence from cross-camera association
                best_match_entity.match_confidence = round(best_match_score, 4)
                if curr_obs.plate_text and not best_match_entity.plate_text:
                    best_match_entity.plate_text = curr_obs.plate_text
            else:
                new_gid = f"GV_{next_global_idx:06d}"
                next_global_idx += 1

                new_entity = GlobalVehicleEntity(
                    global_vehicle_id=new_gid,
                    vehicle_class=curr_obs.vehicle_class,
                    observations=[{
                        "camera_id": curr_obs.camera_id,
                        "track_id": curr_obs.track_id,
                    }],
                    plate_text=curr_obs.plate_text,
                    match_confidence=1.0,  # Seed observation confidence
                )
                global_entities.append(new_entity)

        return global_entities, all_explanations


# =============================================================================
# 4. SIMULATION PIPELINE RUNNER
# =============================================================================

def extract_camera_observations(
    video_source: str,
    camera_segments: Dict[str, Dict[str, int]],
    output_dir: Path,
) -> List[CameraVehicleObservation]:
    """
    Runs UnifiedVehiclePipeline on temporal segments of the video to produce
    independent camera observations with separate ByteTrack local trackers.
    """
    all_observations: List[CameraVehicleObservation] = []
    video_path = Path(video_source).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Initialize UnifiedVehiclePipeline once to avoid repeated model loading
    pipeline = UnifiedVehiclePipeline(
        output_dir=str(output_dir),
        vehicle_conf=0.40,
        plate_conf=0.35,
        ocr_interval=5,
        reid_interval=10,
    )

    for cam_id, seg in camera_segments.items():
        start_f = seg["start_frame"]
        end_f = seg["end_frame"]
        logger.info(f"Extracting observations for {cam_id}: frames {start_f} to {end_f}")

        cam_output_dir = output_dir / cam_id
        cam_output_dir.mkdir(parents=True, exist_ok=True)
        cam_json_path = cam_output_dir / "unified_vehicle_tracks.json"

        pipeline.reset()
        pipeline.output_dir = cam_output_dir

        # Process the segment without saving annotated video to maximize execution speed
        seg_result = pipeline.process_video(
            video_source=str(video_path),
            start_frame=start_f,
            end_frame=end_f,
            output_json_path=str(cam_json_path),
            save_video=False,
        )

        tracks = seg_result.get("vehicle_tracks", [])
        logger.info(f"{cam_id} produced {len(tracks)} local vehicle track(s).")

        for t in tracks:
            obs = CameraVehicleObservation(
                camera_id=cam_id,
                track_id=int(t["track_id"]),
                vehicle_class=str(t["vehicle_class"]),
                first_seen_frame=int(t["first_seen_frame"]),
                last_seen_frame=int(t["last_seen_frame"]),
                plate_text=t["plate"]["text"],
                plate_status=str(t["plate"]["status"]),
                plate_confidence=float(t["plate"]["confidence"]),
                representative_embedding=t["reid"]["representative_embedding"],
                last_bbox=t.get("last_bbox"),
                detector_confidence=t.get("detector_confidence"),
                observation_count=int(t.get("reid", {}).get("observation_count", 1)),
                valid_observation_count=int(t.get("plate", {}).get("valid_observation_count", 0)),
                best_ocr_confidence=t.get("plate", {}).get("best_ocr_confidence"),
            )
            all_observations.append(obs)

    return all_observations


def main():
    """CLI execution entrypoint for Checkpoint 6."""
    parser = argparse.ArgumentParser(
        description="NETRA Checkpoint 6: Cross-Camera Re-ID & Global Identity Resolution"
    )
    parser.add_argument(
        "--source", type=str, default="data/videos/test_2.mp4",
        help="Path to source video file (default: data/videos/test_2.mp4)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/cross_camera",
        help="Directory to save cross-camera output JSONs and reports"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
        help=f"Matching decision threshold (default: {DEFAULT_MATCH_THRESHOLD})"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run fast smoke test on 75 total frames (25 frames per segment)"
    )
    parser.add_argument(
        "--reuse-observations", action="store_true",
        help="Reuse existing cross_camera_observations.json if available"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_json_path = output_dir / "cross_camera_observations.json"
    entities_json_path = output_dir / "global_vehicle_entities.json"
    matches_json_path = output_dir / "cross_camera_matches.json"
    report_path = output_dir / "cross_camera_report.md"

    segments = SMOKE_CAMERA_SEGMENTS if args.smoke_test else DEFAULT_CAMERA_SEGMENTS
    logger.info("=" * 70)
    logger.info("NETRA CHECKPOINT 6: CROSS-CAMERA VEHICLE RE-IDENTIFICATION")
    logger.info("=" * 70)
    logger.info(f"Mode: {'SMOKE TEST' if args.smoke_test else 'FULL SIMULATION'}")
    logger.info(f"Camera Segments: {segments}")

    t_start = time.perf_counter()

    # Step 1: Extract or Load Camera Observations
    if args.reuse_observations and obs_json_path.exists():
        logger.info(f"Loading existing observations from: {obs_json_path}")
        with open(obs_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        observations = [
            CameraVehicleObservation(**item) for item in data["camera_observations"]
        ]
    else:
        observations = extract_camera_observations(
            video_source=args.source,
            camera_segments=segments,
            output_dir=output_dir,
        )
        # Save cross_camera_observations.json
        obs_payload = {
            "metadata": {
                "source_video": str(Path(args.source).resolve()),
                "camera_segments": segments,
                "total_observations": len(observations),
                "is_simulation": True,
                "notice": (
                    "Cross-camera identity resolution is algorithmically validated using "
                    "simulated camera segments from one source video. Real-world cross-camera "
                    "accuracy requires synchronized multi-camera ground-truth data."
                ),
            },
            "camera_observations": [obs.to_dict() for obs in observations],
        }
        with open(obs_json_path, "w", encoding="utf-8") as f:
            json.dump(obs_payload, f, indent=2)
        logger.info(f"Saved observations JSON: {obs_json_path}")

    # Step 2: Run Cross-Camera Matcher & Global Entity Resolution
    matcher = CrossCameraMatcher(match_threshold=args.threshold)
    global_entities, match_explanations = matcher.resolve_global_identities(observations)

    # Step 3: Compile and Save Match Records
    matches_payload = {
        "metadata": {
            "decision_threshold": args.threshold,
            "weights": {
                "reid": WEIGHT_REID,
                "plate": WEIGHT_PLATE,
                "class": WEIGHT_CLASS,
            },
            "total_candidate_comparisons": len(match_explanations),
            "accepted_matches": sum(1 for m in match_explanations if m.matched),
            "rejected_matches": sum(1 for m in match_explanations if not m.matched),
        },
        "matches": [m.to_dict() for m in match_explanations],
    }
    with open(matches_json_path, "w", encoding="utf-8") as f:
        json.dump(matches_payload, f, indent=2)
    logger.info(f"Saved matches JSON: {matches_json_path}")

    # Step 4: Compile and Save Global Vehicle Entities
    entities_payload = {
        "metadata": {
            "total_global_entities": len(global_entities),
            "multi_camera_entities": sum(1 for e in global_entities if len(e.observations) > 1),
            "single_camera_entities": sum(1 for e in global_entities if len(e.observations) == 1),
            "entities_with_plate": sum(1 for e in global_entities if e.plate_text is not None),
        },
        "global_vehicles": [e.to_dict() for e in global_entities],
    }
    with open(entities_json_path, "w", encoding="utf-8") as f:
        json.dump(entities_payload, f, indent=2)
    logger.info(f"Saved global entities JSON: {entities_json_path}")

    t_total = time.perf_counter() - t_start

    # Summary Statistics
    accepted_count = sum(1 for m in match_explanations if m.matched)
    rejected_count = sum(1 for m in match_explanations if not m.matched)
    multi_cam_count = sum(1 for e in global_entities if len(e.observations) > 1)

    print("\n" + "=" * 70)
    print("NETRA CROSS-CAMERA MATCHING EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Observations:          {len(observations)}")
    print(f"Candidate Comparisons:       {len(match_explanations)}")
    print(f"Accepted Matches:            {accepted_count}")
    print(f"Rejected Matches:            {rejected_count}")
    print(f"Global Vehicle Entities:     {len(global_entities)}")
    print(f"Multi-Camera Linked Entities:{multi_cam_count}")
    print(f"Execution Runtime:           {t_total:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
