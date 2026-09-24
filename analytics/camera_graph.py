"""
NETRA — Camera Network Graph & Transition Probabilities
======================================================
Module: analytics/camera_graph.py

Description:
    Constructs a directed graph representation of the camera surveillance network.
    Derives empirical route transition probabilities P(next_camera | current_camera)
    from observed multi-camera vehicle trajectories.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [NETRA.CameraGraph] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NETRA.CameraGraph")

DEFAULT_CAMERAS_CONFIG = "configs/cameras.yaml"
DEFAULT_SEGMENTS_JSON = "runs/trajectory/trajectory_segments.json"
DEFAULT_OUTPUT_DIR = "runs/analytics"


@dataclass
class CameraGraphNode:
    camera_id: str
    latitude: float
    longitude: float
    road_name: str
    speed_limit_kmh: float
    description: str = ""


@dataclass
class CameraGraphEdge:
    from_camera: str
    to_camera: str
    distance_meters: float
    road_name: str
    speed_limit_kmh: float
    observed_vehicle_count: int
    bearing_degrees: float
    direction: str


@dataclass
class RouteTransitionProbability:
    from_camera: str
    to_camera: str
    transition_count: int
    total_outgoing_count: int
    probability: float


class CameraNetworkGraph:
    """
    Directed graph model of the camera network with transition probability estimation.
    """

    def __init__(self, cameras_config_path: str = DEFAULT_CAMERAS_CONFIG):
        self.nodes: Dict[str, CameraGraphNode] = self._load_camera_nodes(cameras_config_path)
        self.edges: Dict[Tuple[str, str], CameraGraphEdge] = {}
        self.transition_probs: Dict[str, List[RouteTransitionProbability]] = {}

    def _load_camera_nodes(self, config_path: str) -> Dict[str, CameraGraphNode]:
        path = Path(config_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Cameras config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cam_dict = data.get("cameras", {})
        nodes = {}
        for cid, cfg in cam_dict.items():
            nodes[cid] = CameraGraphNode(
                camera_id=cid,
                latitude=float(cfg["latitude"]),
                longitude=float(cfg["longitude"]),
                road_name=str(cfg.get("road_name", "UNKNOWN")),
                speed_limit_kmh=float(cfg.get("speed_limit_kmh", 40.0)),
                description=str(cfg.get("description", "")),
            )
        return nodes

    def build_from_segments(self, segments_json_path: str) -> None:
        """
        Builds graph edges and empirical transition probabilities from trajectory segments.
        """
        seg_file = Path(segments_json_path).resolve()
        if not seg_file.exists():
            raise FileNotFoundError(f"Segments JSON not found: {seg_file}")

        with open(seg_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("trajectory_segments", [])
        logger.info(f"Processing {len(segments)} segments to build camera network graph...")

        # Aggregate counts and metadata per directed camera pair (from_cam, to_cam)
        edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for s in segments:
            u = s["from_camera"]
            v = s["to_camera"]
            pair = (u, v)

            if pair not in edge_data:
                edge_data[pair] = {
                    "count": 0,
                    "distance_meters": float(s["distance_meters"]),
                    "bearing_degrees": float(s["bearing_degrees"]),
                    "direction": str(s["direction"]),
                    "road_name": f"{self.nodes[u].road_name} -> {self.nodes[v].road_name}",
                    "speed_limit_kmh": float(s["speed_limit_kmh"]),
                }
            edge_data[pair]["count"] += 1

        # Instantiate CameraGraphEdge objects
        self.edges.clear()
        for (u, v), d in edge_data.items():
            self.edges[(u, v)] = CameraGraphEdge(
                from_camera=u,
                to_camera=v,
                distance_meters=round(d["distance_meters"], 2),
                road_name=d["road_name"],
                speed_limit_kmh=round(d["speed_limit_kmh"], 1),
                observed_vehicle_count=d["count"],
                bearing_degrees=round(d["bearing_degrees"], 2),
                direction=d["direction"],
            )

        # Compute empirical transition probabilities P(v | u)
        # Sum outgoing transitions for each source camera u
        outgoing_totals: Dict[str, int] = {}
        for (u, v), edge in self.edges.items():
            outgoing_totals[u] = outgoing_totals.get(u, 0) + edge.observed_vehicle_count

        self.transition_probs.clear()
        for u in self.nodes:
            self.transition_probs[u] = []
            total_out = outgoing_totals.get(u, 0)
            if total_out > 0:
                for (src, dst), edge in self.edges.items():
                    if src == u:
                        prob = edge.observed_vehicle_count / float(total_out)
                        self.transition_probs[u].append(
                            RouteTransitionProbability(
                                from_camera=u,
                                to_camera=dst,
                                transition_count=edge.observed_vehicle_count,
                                total_outgoing_count=total_out,
                                probability=round(prob, 4),
                            )
                        )
                # Sort descending by probability
                self.transition_probs[u].sort(key=lambda x: x.probability, reverse=True)

        logger.info(f"Constructed graph with {len(self.nodes)} nodes and {len(self.edges)} directed edges.")

    def export_graph_json(self, output_path: str) -> None:
        """Exports the camera network graph structure to JSON."""
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        graph_dict = {
            "graph_type": "camera_network_directed_graph",
            "description": "Empirical camera network graph based on observed vehicle trajectories",
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(graph_dict, f, indent=2)
        logger.info(f"Exported camera network graph JSON: {p}")

    def export_transition_probabilities_json(self, output_path: str) -> None:
        """Exports transition probabilities P(next | current) to JSON."""
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        flat_list = []
        structured_dict = {}

        for u, trans_list in self.transition_probs.items():
            structured_dict[u] = []
            for t in trans_list:
                flat_list.append(asdict(t))
                structured_dict[u].append({
                    "to_camera": t.to_camera,
                    "probability": t.probability,
                    "count": t.transition_count,
                })

        output_data = {
            "formula": "P(next | current) = count(current -> next) / total_outgoing(current)",
            "camera_transition_probabilities": structured_dict,
            "transitions_list": flat_list,
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Exported transition probabilities JSON: {p}")


def main():
    parser = argparse.ArgumentParser(description="Build NETRA Camera Network Graph & Transition Probabilities")
    parser.add_argument("--cameras-config", type=str, default=DEFAULT_CAMERAS_CONFIG)
    parser.add_argument("--segments-json", type=str, default=DEFAULT_SEGMENTS_JSON)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = CameraNetworkGraph(cameras_config_path=args.cameras_config)
    graph.build_from_segments(segments_json_path=args.segments_json)

    graph.export_graph_json(str(out_dir / "camera_network_graph.json"))
    graph.export_transition_probabilities_json(str(out_dir / "route_transition_probabilities.json"))


if __name__ == "__main__":
    main()
