"""
NETRA — Empirical Route Prediction Foundation
=============================================
Module: analytics/route_prediction.py

Description:
    Implements a baseline next-camera and vehicle-specific route prediction engine
    based strictly on empirical transition probabilities observed across multi-camera
    trajectories.

Scope Boundaries:
    - This is a historical transition baseline, NOT a deep learning model or LLM.
    - No fake speed values or synthetic transit times are used as prediction features.
    - Returns 'prediction_available: false' when insufficient historical evidence exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [NETRA.RoutePredictor] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NETRA.RoutePredictor")

DEFAULT_TRANSITIONS_JSON = "runs/analytics/route_transition_probabilities.json"
DEFAULT_TRAJECTORIES_JSON = "runs/trajectory/vehicle_trajectories.json"
DEFAULT_OUTPUT_DIR = "runs/analytics"


@dataclass
class NextCameraCandidate:
    next_camera: str
    probability: float
    observed_historical_transitions: int


@dataclass
class VehicleRoutePrediction:
    global_vehicle_id: str
    current_camera: str
    vehicle_class: str
    plate_text: Optional[str]
    prediction_available: bool
    predictions: List[Dict[str, Any]]
    reason: Optional[str] = None


class RoutePredictor:
    """
    Empirical Markovian route predictor using observed camera-to-camera transitions.
    """

    def __init__(self, transitions_json_path: str = DEFAULT_TRANSITIONS_JSON):
        self.transitions = self._load_transition_probabilities(transitions_json_path)

    @staticmethod
    def _load_transition_probabilities(path_str: str) -> Dict[str, List[Dict[str, Any]]]:
        p = Path(path_str).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Transitions probabilities JSON not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("camera_transition_probabilities", {})

    def predict_next_camera(self, current_camera_id: str) -> List[NextCameraCandidate]:
        """
        Ranks candidate next cameras given current camera location based on empirical transitions.
        """
        candidates_raw = self.transitions.get(current_camera_id, [])
        if not candidates_raw:
            return []

        ranked: List[NextCameraCandidate] = []
        for c in candidates_raw:
            ranked.append(
                NextCameraCandidate(
                    next_camera=c["to_camera"],
                    probability=float(c["probability"]),
                    observed_historical_transitions=int(c.get("count", 0)),
                )
            )
        ranked.sort(key=lambda x: x.probability, reverse=True)
        return ranked

    def predict_vehicle_route(
        self,
        global_vehicle_id: str,
        current_camera_id: str,
        vehicle_class: str,
        plate_text: Optional[str] = None,
    ) -> VehicleRoutePrediction:
        """
        Predicts the next probable camera node for a specific global vehicle entity.
        Returns prediction_available=False if current camera has no outgoing historical transitions.
        """
        candidates = self.predict_next_camera(current_camera_id)

        if not candidates:
            return VehicleRoutePrediction(
                global_vehicle_id=global_vehicle_id,
                current_camera=current_camera_id,
                vehicle_class=vehicle_class,
                plate_text=plate_text,
                prediction_available=False,
                predictions=[],
                reason="insufficient historical transition data",
            )

        return VehicleRoutePrediction(
            global_vehicle_id=global_vehicle_id,
            current_camera=current_camera_id,
            vehicle_class=vehicle_class,
            plate_text=plate_text,
            prediction_available=True,
            predictions=[asdict(c) for c in candidates],
            reason=None,
        )

    def generate_all_predictions(
        self,
        trajectories_json_path: str,
        output_json_path: str,
    ) -> List[VehicleRoutePrediction]:
        """
        Generates route predictions for all active vehicle entities from their last-known camera.
        """
        traj_file = Path(trajectories_json_path).resolve()
        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectories JSON not found: {traj_file}")

        with open(traj_file, "r", encoding="utf-8") as f:
            trajs = json.load(f).get("vehicle_trajectories", [])

        all_predictions: List[VehicleRoutePrediction] = []
        for t in trajs:
            gid = t["global_vehicle_id"]
            vcls = t["vehicle_class"]
            vplate = t.get("plate_text")
            cam_seq = t.get("camera_sequence", [])

            if not cam_seq:
                continue

            # Predict next step from the vehicle's last-known camera observation
            current_cam = cam_seq[-1]
            pred = self.predict_vehicle_route(
                global_vehicle_id=gid,
                current_camera_id=current_cam,
                vehicle_class=vcls,
                plate_text=vplate,
            )
            all_predictions.append(pred)

        out_p = Path(output_json_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "predictor_type": "empirical_historical_transition_baseline",
            "description": "Baseline next-camera predictor based on observed transition frequencies",
            "total_entities_evaluated": len(all_predictions),
            "predictions_available_count": sum(1 for p in all_predictions if p.prediction_available),
            "predictions_unavailable_count": sum(1 for p in all_predictions if not p.prediction_available),
            "vehicle_route_predictions": [asdict(p) for p in all_predictions],
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Saved route predictions to: {out_p}")
        return all_predictions


def main():
    parser = argparse.ArgumentParser(description="NETRA Route Prediction Foundation")
    parser.add_argument("--transitions-json", type=str, default=DEFAULT_TRANSITIONS_JSON)
    parser.add_argument("--trajectories-json", type=str, default=DEFAULT_TRAJECTORIES_JSON)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = RoutePredictor(transitions_json_path=args.transitions_json)
    predictions = predictor.generate_all_predictions(
        trajectories_json_path=args.trajectories_json,
        output_json_path=str(out_dir / "route_predictions.json"),
    )

    avail_count = sum(1 for p in predictions if p.prediction_available)
    unavail_count = len(predictions) - avail_count

    print("\n" + "=" * 70)
    print("NETRA ROUTE PREDICTION EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Entities Evaluated:      {len(predictions)}")
    print(f"Predictions Available:         {avail_count}")
    print(f"Predictions Unavailable:       {unavail_count} (Corridor terminus nodes)")
    print(f"Output JSON:                   {out_dir / 'route_predictions.json'}")

    # Show example predictions
    sample_avail = next((p for p in predictions if p.prediction_available), None)
    sample_unavail = next((p for p in predictions if not p.prediction_available), None)

    if sample_avail:
        print(f"\n[Example Available Prediction]")
        print(f"  Vehicle ID:     {sample_avail.global_vehicle_id} ({sample_avail.vehicle_class})")
        print(f"  Current Camera: {sample_avail.current_camera}")
        for c in sample_avail.predictions:
            print(f"  -> Next: {c['next_camera']} | Probability: {c['probability'] * 100:.1f}% ({c['observed_historical_transitions']} obs)")

    if sample_unavail:
        print(f"\n[Example Unavailable Prediction]")
        print(f"  Vehicle ID:     {sample_unavail.global_vehicle_id} ({sample_unavail.vehicle_class})")
        print(f"  Current Camera: {sample_unavail.current_camera}")
        print(f"  Status:         Unavailable ({sample_unavail.reason})")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
