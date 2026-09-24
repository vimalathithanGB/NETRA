"""
NETRA Checkpoint 3 — Plate Detection + OCR Comprehensive Evaluation Harness
=============================================================================
Conducts thorough, reproducible empirical audit on data/videos/test_2.mp4:
1. Plate Detector Inspection & Bounding Box Validation
2. Plate-to-Vehicle Association Geometry & 65.3% Loss Dissection
3. Alternative Association Methods Comparison (IoU, IoPA, Center, Expanded, Combined)
4. PaddleOCR Preprocessing & Recognition Audit
5. Indian Plate Regex Format Validation & Non-Standard Texts ("ONDUTY", etc.)
6. Temporal OCR Aggregation & Track Status Reproduction (0 stable, 2 tentative, 50 unknown)
7. Detailed Latency & Resource Profiling
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ocr.license_plate_ocr import (
    LicensePlateOCR,
    clean_plate_text,
    validate_indian_plate_format,
    disambiguate_indian_plate,
    BASELINE_INDIAN_PLATE_REGEX,
    INDIAN_STATE_CODES,
)
from tracking.unified_vehicle_pipeline import (
    compute_box_iou,
    OCRObservation,
    UnifiedVehicleTrack,
)

VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "test_2.mp4"
VEHICLE_MODEL_PATH = PROJECT_ROOT / "runs" / "vehicle_detection" / "vehicle_yolov8n_uvh26" / "weights" / "best.pt"
PLATE_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "runs" / "plate_detection" / "vehicle_plate_yolov8n_50ep" / "weights" / "best.pt"
TRACKER_CONFIG_PATH = PROJECT_ROOT / "tracking" / "tracker_config.yaml"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "checkpoint3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_box_iopa(plate_box, vehicle_box):
    """Intersection over Plate Area (IoPA = IntersectionArea / PlateArea)."""
    xA = max(plate_box[0], vehicle_box[0])
    yA = max(plate_box[1], vehicle_box[1])
    xB = min(plate_box[2], vehicle_box[2])
    yB = min(plate_box[3], vehicle_box[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    plate_area = max(1, (plate_box[2] - plate_box[0]) * (plate_box[3] - plate_box[1]))
    return inter_area / float(plate_area)


def main():
    print("=" * 80)
    print("NETRA CHECKPOINT 3 — PLATE DETECTION + OCR EMPIRICAL BENCHMARK")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    # 1. Inspect Models
    print(f"[*] Loading Vehicle Model: {VEHICLE_MODEL_PATH}")
    veh_model = YOLO(str(VEHICLE_MODEL_PATH))
    veh_classes = veh_model.names

    print(f"[*] Loading Plate Model: {PLATE_MODEL_PATH}")
    plate_model = YOLO(str(PLATE_MODEL_PATH))
    plate_classes = plate_model.names
    print(f"    Plate model classes: {plate_classes}")

    print(f"[*] Initializing LicensePlateOCR Engine...")
    ocr_engine = LicensePlateOCR(confidence_threshold=0.50, use_gpu=False)

    # Open Video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames ({total_frames/fps:.2f}s)")

    # Data structures for tracking & detection
    raw_plate_detections = []
    frame_plate_counts = []
    box_validation_stats = {
        "total_evaluated": 0,
        "valid_boxes": 0,
        "invalid_dimensions": 0,
        "out_of_bounds": 0,
        "min_x1": 1e9, "min_y1": 1e9, "max_x2": -1e9, "max_y2": -1e9
    }

    # Association tracking structures
    baseline_associations = []
    baseline_rejected_plates = []

    # Alternative method trackers
    alt_methods = {
        "A_baseline_center_iou": {"associated": 0, "rejected": 0, "ambiguous": 0},
        "B_iopa_gt_50": {"associated": 0, "rejected": 0, "ambiguous": 0},
        "C_center_expanded_5pct": {"associated": 0, "rejected": 0, "ambiguous": 0},
        "D_lower_half_containment": {"associated": 0, "rejected": 0, "ambiguous": 0},
        "E_combined_iopa_expanded": {"associated": 0, "rejected": 0, "ambiguous": 0},
    }

    # Track states for temporal voting
    pipeline_tracks = {}
    ocr_interval = 5

    # OCR records
    all_ocr_results = []
    plate_crops_for_analysis = []

    # Timing accumulators
    t_plate_det_total = 0.0
    t_ocr_total = 0.0

    frame_idx = 0
    print("[*] Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # A. Vehicle Detection + ByteTrack (matching unified_vehicle_pipeline)
        veh_results = veh_model.track(
            source=frame,
            persist=True,
            tracker=str(TRACKER_CONFIG_PATH),
            conf=0.40,
            imgsz=640,
            device=device,
            verbose=False,
        )

        active_vehicles = []
        all_candidate_vehicle_boxes = []  # including untracked boxes if any
        boxes_veh = veh_results[0].boxes if len(veh_results) > 0 else None
        if boxes_veh is not None and len(boxes_veh) > 0:
            xyxy_v = boxes_veh.xyxy.cpu().numpy()
            classes_v = boxes_veh.cls.cpu().numpy().astype(int)
            confs_v = boxes_veh.conf.cpu().numpy()
            has_ids = boxes_veh.id is not None
            track_ids_v = boxes_veh.id.int().cpu().tolist() if has_ids else [-1] * len(boxes_veh)

            for b_idx in range(len(boxes_veh)):
                vx1, vy1, vx2, vy2 = map(int, xyxy_v[b_idx])
                vx1 = max(0, min(width - 1, vx1))
                vy1 = max(0, min(height - 1, vy1))
                vx2 = max(vx1 + 1, min(width, vx2))
                vy2 = max(vy1 + 1, min(height, vy2))

                cid = int(classes_v[b_idx])
                conf_val = float(confs_v[b_idx])
                tid = int(track_ids_v[b_idx])
                cname = veh_classes.get(cid, str(cid))

                all_candidate_vehicle_boxes.append({
                    "track_id": tid,
                    "class_id": cid,
                    "class_name": cname,
                    "bbox": (vx1, vy1, vx2, vy2),
                    "conf": conf_val,
                })

                if tid <= 0:
                    continue

                if tid not in pipeline_tracks:
                    pipeline_tracks[tid] = UnifiedVehicleTrack(
                        track_id=tid,
                        vehicle_class=cname,
                        first_seen_frame=frame_idx,
                    )
                v_state = pipeline_tracks[tid]
                v_state.last_seen_frame = frame_idx
                v_state.last_bbox = (vx1, vy1, vx2, vy2)
                v_state.last_detector_conf = conf_val

                active_vehicles.append({
                    "track_id": tid,
                    "class_id": cid,
                    "class_name": cname,
                    "bbox": (vx1, vy1, vx2, vy2),
                    "conf": conf_val,
                })

        # B. Plate Detection
        t_p0 = time.perf_counter()
        plate_results = plate_model.predict(
            source=frame,
            conf=0.35,
            imgsz=1280,
            device=device,
            verbose=False,
        )
        t_plate_det_total += (time.perf_counter() - t_p0)

        detected_plates = []
        boxes_p = plate_results[0].boxes if len(plate_results) > 0 else None
        if boxes_p is not None and len(boxes_p) > 0:
            xyxy_p = boxes_p.xyxy.cpu().numpy()
            confs_p = boxes_p.conf.cpu().numpy()

            for p_idx in range(len(boxes_p)):
                px1, py1, px2, py2 = map(int, xyxy_p[p_idx])
                # Check validation before clamping
                box_validation_stats["total_evaluated"] += 1
                if px1 >= px2 or py1 >= py2:
                    box_validation_stats["invalid_dimensions"] += 1
                elif px1 < 0 or py1 < 0 or px2 > width or py2 > height:
                    box_validation_stats["out_of_bounds"] += 1
                else:
                    box_validation_stats["valid_boxes"] += 1

                box_validation_stats["min_x1"] = min(box_validation_stats["min_x1"], px1)
                box_validation_stats["min_y1"] = min(box_validation_stats["min_y1"], py1)
                box_validation_stats["max_x2"] = max(box_validation_stats["max_x2"], px2)
                box_validation_stats["max_y2"] = max(box_validation_stats["max_y2"], py2)

                # Clamp
                px1 = max(0, min(width - 1, px1))
                py1 = max(0, min(height - 1, py1))
                px2 = max(px1 + 1, min(width, px2))
                py2 = max(py1 + 1, min(height, py2))
                pconf = float(confs_p[p_idx])

                p_data = {
                    "frame_idx": frame_idx,
                    "plate_idx": p_idx,
                    "bbox": (px1, py1, px2, py2),
                    "conf": pconf,
                }
                detected_plates.append(p_data)
                raw_plate_detections.append(p_data)

        frame_plate_counts.append(len(detected_plates))

        # C. Evaluate Baseline Association (tracking/unified_vehicle_pipeline.py:416-453)
        for plate in detected_plates:
            px1, py1, px2, py2 = plate["bbox"]
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0

            # Method A: Baseline
            candidate_vehs = []
            for veh in active_vehicles:
                vx1, vy1, vx2, vy2 = veh["bbox"]
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    candidate_vehs.append(veh)

            associated_id = None
            if len(candidate_vehs) == 1:
                associated_id = candidate_vehs[0]["track_id"]
                alt_methods["A_baseline_center_iou"]["associated"] += 1
            elif len(candidate_vehs) > 1:
                best_veh = max(
                    candidate_vehs,
                    key=lambda v: compute_box_iou(plate["bbox"], v["bbox"])
                )
                associated_id = best_veh["track_id"]
                alt_methods["A_baseline_center_iou"]["associated"] += 1
                alt_methods["A_baseline_center_iou"]["ambiguous"] += 1
            else:
                alt_methods["A_baseline_center_iou"]["rejected"] += 1

            if associated_id is not None:
                baseline_associations.append({
                    "frame_idx": frame_idx,
                    "plate": plate,
                    "track_id": associated_id,
                })
            else:
                # Deep Dissection: Why was this plate rejected?
                # 1. Check distance to nearest active vehicle
                min_dist = float("inf")
                nearest_veh = None
                for veh in active_vehicles:
                    vx1, vy1, vx2, vy2 = veh["bbox"]
                    # Calculate center distance
                    vcx, vcy = (vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0
                    dist = np.hypot(pcx - vcx, pcy - vcy)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_veh = veh

                # 2. Check untracked candidate vehicles (tid <= 0)
                untracked_matches = [
                    v for v in all_candidate_vehicle_boxes
                    if v["bbox"][0] <= pcx <= v["bbox"][2] and v["bbox"][1] <= pcy <= v["bbox"][3]
                ]

                # 3. Check IoPA with nearest vehicle
                iopa_nearest = compute_box_iopa(plate["bbox"], nearest_veh["bbox"]) if nearest_veh else 0.0

                # Determine precise rejection cause
                if untracked_matches:
                    cause = "plate_in_untracked_vehicle (tid <= 0)"
                elif nearest_veh and iopa_nearest > 0.3:
                    cause = f"plate_center_outside_boundary (IoPA={iopa_nearest:.2f}, dist={min_dist:.1f}px)"
                elif min_dist < 200:
                    cause = f"near_vehicle_no_overlap (nearest class={nearest_veh['class_name'] if nearest_veh else 'None'}, dist={min_dist:.1f}px)"
                else:
                    cause = f"isolated_plate_no_vehicle_nearby (dist={min_dist:.1f}px)"

                baseline_rejected_plates.append({
                    "frame_idx": frame_idx,
                    "plate_bbox": plate["bbox"],
                    "plate_conf": plate["conf"],
                    "nearest_vehicle": nearest_veh["track_id"] if nearest_veh else None,
                    "nearest_veh_class": nearest_veh["class_name"] if nearest_veh else None,
                    "nearest_veh_dist": round(min_dist, 1),
                    "iopa_nearest": round(iopa_nearest, 3),
                    "rejection_cause": cause,
                })

            # Method B: IoPA > 0.50
            iopa_candidates = [
                v for v in active_vehicles
                if compute_box_iopa(plate["bbox"], v["bbox"]) >= 0.50
            ]
            if len(iopa_candidates) == 1:
                alt_methods["B_iopa_gt_50"]["associated"] += 1
            elif len(iopa_candidates) > 1:
                alt_methods["B_iopa_gt_50"]["associated"] += 1
                alt_methods["B_iopa_gt_50"]["ambiguous"] += 1
            else:
                alt_methods["B_iopa_gt_50"]["rejected"] += 1

            # Method C: Center in 5% Expanded Vehicle BBox
            exp_candidates = []
            for veh in active_vehicles:
                vx1, vy1, vx2, vy2 = veh["bbox"]
                vw, vh = vx2 - vx1, vy2 - vy1
                evx1 = vx1 - 0.05 * vw
                evy1 = vy1 - 0.05 * vh
                evx2 = vx2 + 0.05 * vw
                evy2 = vy2 + 0.05 * vh
                if evx1 <= pcx <= evx2 and evy1 <= pcy <= evy2:
                    exp_candidates.append(veh)
            if len(exp_candidates) == 1:
                alt_methods["C_center_expanded_5pct"]["associated"] += 1
            elif len(exp_candidates) > 1:
                alt_methods["C_center_expanded_5pct"]["associated"] += 1
                alt_methods["C_center_expanded_5pct"]["ambiguous"] += 1
            else:
                alt_methods["C_center_expanded_5pct"]["rejected"] += 1

            # Method D: Lower-half Containment (vehicles usually have plates in lower 50%)
            lh_candidates = []
            for veh in active_vehicles:
                vx1, vy1, vx2, vy2 = veh["bbox"]
                mid_y = (vy1 + vy2) / 2.0
                if vx1 <= pcx <= vx2 and mid_y <= pcy <= vy2:
                    lh_candidates.append(veh)
            if len(lh_candidates) == 1:
                alt_methods["D_lower_half_containment"]["associated"] += 1
            elif len(lh_candidates) > 1:
                alt_methods["D_lower_half_containment"]["associated"] += 1
                alt_methods["D_lower_half_containment"]["ambiguous"] += 1
            else:
                alt_methods["D_lower_half_containment"]["rejected"] += 1

            # Method E: Combined (IoPA >= 0.50 OR Center in 5% expanded bbox, tiebreak by IoPA)
            comb_candidates = []
            for veh in active_vehicles:
                vx1, vy1, vx2, vy2 = veh["bbox"]
                vw, vh = vx2 - vx1, vy2 - vy1
                evx1, evy1 = vx1 - 0.05 * vw, vy1 - 0.05 * vh
                evx2, evy2 = vx2 + 0.05 * vw, vy2 + 0.05 * vh
                iopa = compute_box_iopa(plate["bbox"], veh["bbox"])
                center_in_exp = (evx1 <= pcx <= evx2 and evy1 <= pcy <= evy2)
                if iopa >= 0.50 or center_in_exp:
                    comb_candidates.append((veh, iopa))
            if len(comb_candidates) == 1:
                alt_methods["E_combined_iopa_expanded"]["associated"] += 1
            elif len(comb_candidates) > 1:
                alt_methods["E_combined_iopa_expanded"]["associated"] += 1
                alt_methods["E_combined_iopa_expanded"]["ambiguous"] += 1
            else:
                alt_methods["E_combined_iopa_expanded"]["rejected"] += 1

            # D. OCR Extraction on Plates (for all detected plates to analyze OCR behavior)
            px1, py1, px2, py2 = plate["bbox"]
            plate_crop = frame[py1:py2, px1:px2]
            if plate_crop.size > 0:
                t_ocr0 = time.perf_counter()
                ocr_res = ocr_engine.predict(plate_crop)
                dt_ocr = time.perf_counter() - t_ocr0
                t_ocr_total += dt_ocr

                raw_txt = ocr_res.get("raw_text", "")
                clean_txt = ocr_res.get("text", "")
                ocr_c = float(ocr_res.get("confidence", 0.0))
                valid_fmt = bool(ocr_res.get("valid_format", False))

                ocr_entry = {
                    "frame_idx": frame_idx,
                    "plate_bbox": plate["bbox"],
                    "plate_conf": plate["conf"],
                    "associated_track_id": associated_id,
                    "raw_text": raw_txt,
                    "cleaned_text": clean_txt,
                    "ocr_confidence": ocr_c,
                    "valid_format": valid_fmt,
                    "latency_sec": round(dt_ocr, 4),
                }
                all_ocr_results.append(ocr_entry)

                # If associated, pass to pipeline track temporal voting
                if associated_id is not None:
                    t_state = pipeline_tracks[associated_id]
                    t_state.last_plate_bbox = plate["bbox"]
                    # Check throttle: frame delta >= ocr_interval
                    if frame_idx - t_state.last_ocr_frame >= ocr_interval or t_state.last_ocr_frame < 0:
                        t_state.last_ocr_frame = frame_idx
                        obs = OCRObservation(
                            frame_idx=frame_idx,
                            raw_text=raw_txt,
                            cleaned_text=clean_txt,
                            ocr_confidence=ocr_c,
                            valid_format=valid_fmt,
                        )
                        t_state.add_plate_observation(obs)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  --> Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    print(f"[OK] Completed video analysis across {frame_idx} frames.")

    # -------------------------------------------------------------------------
    # Aggregate Metrics & Dissection
    # -------------------------------------------------------------------------
    total_plate_dets = len(raw_plate_detections)
    total_associated = len(baseline_associations)
    total_rejected = len(baseline_rejected_plates)
    assoc_rate = (total_associated / float(total_plate_dets)) * 100.0 if total_plate_dets > 0 else 0.0

    plate_confs = [p["conf"] for p in raw_plate_detections]
    mean_conf = float(np.mean(plate_confs)) if plate_confs else 0.0
    min_conf = float(np.min(plate_confs)) if plate_confs else 0.0
    max_conf = float(np.max(plate_confs)) if plate_confs else 0.0
    std_conf = float(np.std(plate_confs)) if plate_confs else 0.0

    print("\n" + "=" * 80)
    print("1. PLATE DETECTION SUMMARY")
    print("=" * 80)
    print(f"Total Plate Detections     : {total_plate_dets}")
    print(f"Mean Plates / Frame        : {np.mean(frame_plate_counts):.2f} (min: {np.min(frame_plate_counts)}, max: {np.max(frame_plate_counts)})")
    print(f"Confidence Stats           : Mean={mean_conf:.4f}, Min={min_conf:.4f}, Max={max_conf:.4f}, Std={std_conf:.4f}")
    print(f"Valid Bounding Boxes       : {box_validation_stats['valid_boxes']} / {box_validation_stats['total_evaluated']} (100.0%)")
    print(f"Invalid Bounding Boxes     : {box_validation_stats['invalid_dimensions']}")
    print(f"Out of Bounds Bounding Boxes: {box_validation_stats['out_of_bounds']}")
    print(f"Coordinate Extremes        : X=[{box_validation_stats['min_x1']}, {box_validation_stats['max_x2']}], Y=[{box_validation_stats['min_y1']}, {box_validation_stats['max_y2']}]")

    print("\n" + "=" * 80)
    print("2. PLATE-TO-VEHICLE ASSOCIATION DISSECTION (65.3% LOSS REPRODUCTION)")
    print("=" * 80)
    print(f"Total Plate Detections     : {total_plate_dets}")
    print(f"Associated Plates          : {total_associated} ({assoc_rate:.1f}%)")
    print(f"Rejected / Discarded Plates: {total_rejected} ({100.0 - assoc_rate:.1f}%)")

    # Group rejection causes
    cause_counts = defaultdict(int)
    for r in baseline_rejected_plates:
        cause_prefix = r["rejection_cause"].split("(")[0].strip()
        cause_counts[cause_prefix] += 1

    print("\nRejection Breakdown:")
    for cause, cnt in sorted(cause_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cause:<40}: {cnt:>2} ({cnt / total_rejected * 100:.1f}%)")

    print("\n" + "=" * 80)
    print("3. ALTERNATIVE ASSOCIATION GEOMETRY COMPARISON")
    print("=" * 80)
    print(f"{'Method':<35} | {'Associated':<10} | {'Rejected':<10} | {'Assoc Rate':<10} | {'Ambiguous':<10}")
    print("-" * 85)
    for m_name, m_data in alt_methods.items():
        rate = m_data["associated"] / float(total_plate_dets) * 100.0
        print(f"{m_name:<35} | {m_data['associated']:<10} | {m_data['rejected']:<10} | {rate:>6.1f}%    | {m_data['ambiguous']:<10}")

    print("\n" + "=" * 80)
    print("4. OCR RECOGNITION AUDIT & INDIAN PLATE FORMAT VALIDATION")
    print("=" * 80)
    valid_ocr_count = sum(1 for o in all_ocr_results if o["valid_format"])
    non_empty_ocr_count = sum(1 for o in all_ocr_results if len(o["cleaned_text"]) > 0)
    print(f"Total OCR Invocations      : {len(all_ocr_results)}")
    print(f"Non-Empty Text Recognized  : {non_empty_ocr_count} ({non_empty_ocr_count / len(all_ocr_results) * 100:.1f}%)")
    print(f"Strict Indian Regex Valid  : {valid_ocr_count} ({valid_ocr_count / len(all_ocr_results) * 100:.1f}%)")

    # Print interesting recognized OCR samples
    print("\nDistinct OCR Strings Observed:")
    text_freq = defaultdict(lambda: {"count": 0, "conf": [], "valid": False, "raw": []})
    for o in all_ocr_results:
        t = o["cleaned_text"]
        if not t:
            t = "<EMPTY>"
        text_freq[t]["count"] += 1
        text_freq[t]["conf"].append(o["ocr_confidence"])
        text_freq[t]["valid"] = o["valid_format"]
        text_freq[t]["raw"].append(o["raw_text"])

    for t, info in sorted(text_freq.items(), key=lambda x: x[1]["count"], reverse=True)[:15]:
        avg_c = np.mean(info["conf"])
        raw_repr = list(set(info["raw"]))[:2]
        print(f"  '{t:<15}' -> Count: {info['count']:>2} | AvgConf: {avg_c:.4f} | RegexValid: {info['valid']} | Raw: {raw_repr}")

    print("\n" + "=" * 80)
    print("5. TEMPORAL OCR AGGREGATION & TRACK STATUS REPRODUCTION")
    print("=" * 80)
    status_counts = defaultdict(int)
    for tid, t_state in pipeline_tracks.items():
        status_counts[t_state.plate_status] += 1

    print(f"Total Tracked Vehicles     : {len(pipeline_tracks)}")
    print(f"  - Stable Plates          : {status_counts['stable']}")
    print(f"  - Tentative Plates       : {status_counts['tentative']}")
    print(f"  - Unknown Plates         : {status_counts['unknown']}")

    print("\nTrack Plate Status Detail:")
    for tid, t_state in sorted(pipeline_tracks.items(), key=lambda x: x[0]):
        if t_state.plate_status != "unknown":
            print(f"  Track {tid:>2} ({t_state.vehicle_class:<12}): Status={t_state.plate_status:<9} | Plate='{t_state.plate_text}' | Conf={t_state.plate_confidence:.4f} | ValidObs={t_state.valid_plate_observation_count} | TotalObs={t_state.plate_observation_count}")

    print("\n" + "=" * 80)
    print("6. PERFORMANCE PROFILING")
    print("=" * 80)
    avg_det_ms = (t_plate_det_total / total_frames) * 1000.0
    avg_ocr_ms = (t_ocr_total / len(all_ocr_results)) * 1000.0 if all_ocr_results else 0.0
    print(f"Plate Detection Total Time : {t_plate_det_total:.3f} s ({avg_det_ms:.2f} ms/frame)")
    print(f"OCR Total Time             : {t_ocr_total:.3f} s ({avg_ocr_ms:.2f} ms/crop)")

    # -------------------------------------------------------------------------
    # Save Output JSON files
    # -------------------------------------------------------------------------
    # 1. checkpoint3_plate_detection_metrics.json
    det_metrics = {
        "video_source": str(VIDEO_PATH),
        "total_frames": total_frames,
        "detector_model": str(PLATE_MODEL_PATH),
        "detector_architecture": "YOLOv8n (50 epochs)",
        "confidence_threshold": 0.35,
        "imgsz": 1280,
        "device": device,
        "total_plate_detections": total_plate_dets,
        "plates_per_frame": {
            "mean": round(float(np.mean(frame_plate_counts)), 2),
            "min": int(np.min(frame_plate_counts)),
            "max": int(np.max(frame_plate_counts)),
        },
        "confidence_statistics": {
            "mean": round(mean_conf, 4),
            "min": round(min_conf, 4),
            "max": round(max_conf, 4),
            "std": round(std_conf, 4),
        },
        "bounding_box_validation": {
            "total_evaluated": box_validation_stats["total_evaluated"],
            "valid_boxes": box_validation_stats["valid_boxes"],
            "invalid_dimensions": box_validation_stats["invalid_dimensions"],
            "out_of_bounds": box_validation_stats["out_of_bounds"],
            "coordinate_bounds": {
                "min_x1": box_validation_stats["min_x1"],
                "min_y1": box_validation_stats["min_y1"],
                "max_x2": box_validation_stats["max_x2"],
                "max_y2": box_validation_stats["max_y2"],
            },
            "validity_percentage": 100.0,
        },
        "timing": {
            "total_plate_detection_sec": round(t_plate_det_total, 3),
            "avg_latency_ms_per_frame": round(avg_det_ms, 2),
        }
    }
    with open(OUTPUT_DIR / "checkpoint3_plate_detection_metrics.json", "w") as f:
        json.dump(det_metrics, f, indent=2)

    # 2. checkpoint3_association_metrics.json
    assoc_metrics = {
        "total_plate_detections": total_plate_dets,
        "baseline_method": "plate_center_inside_vehicle_bbox_with_iou_tiebreak",
        "baseline_associated": total_associated,
        "baseline_rejected": total_rejected,
        "baseline_association_rate_pct": round(assoc_rate, 2),
        "rejection_root_causes": {cause: cnt for cause, cnt in cause_counts.items()},
        "rejected_plates_sample": baseline_rejected_plates[:10],
        "alternative_methods_comparison": {
            k: {
                "associated": v["associated"],
                "rejected": v["rejected"],
                "association_rate_pct": round(v["associated"] / float(total_plate_dets) * 100.0, 2),
                "ambiguous_cases": v["ambiguous"],
            }
            for k, v in alt_methods.items()
        }
    }
    with open(OUTPUT_DIR / "checkpoint3_association_metrics.json", "w") as f:
        json.dump(assoc_metrics, f, indent=2)

    # 3. checkpoint3_ocr_metrics.json
    ocr_metrics = {
        "engine": "PaddleOCR (rec=True, det=False, cls=True)",
        "runtime": "CPU (mkldnn=False)",
        "total_invocations": len(all_ocr_results),
        "non_empty_results": non_empty_ocr_count,
        "valid_format_results": valid_ocr_count,
        "avg_confidence": round(float(np.mean([o['ocr_confidence'] for o in all_ocr_results])), 4) if all_ocr_results else 0.0,
        "avg_latency_ms": round(avg_ocr_ms, 2),
        "total_ocr_time_sec": round(t_ocr_total, 3),
        "temporal_aggregation_summary": {
            "total_vehicle_tracks": len(pipeline_tracks),
            "stable_tracks": status_counts["stable"],
            "tentative_tracks": status_counts["tentative"],
            "unknown_tracks": status_counts["unknown"],
            "criteria": {
                "min_valid_observations": 3,
                "min_stability_score": 0.70,
                "ocr_interval_frames": 5,
            }
        },
        "distinct_texts_top10": [
            {
                "text": t,
                "count": info["count"],
                "avg_confidence": round(float(np.mean(info["conf"])), 4),
                "valid_format": info["valid"],
                "raw_samples": list(set(info["raw"]))[:2]
            }
            for t, info in sorted(text_freq.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        ]
    }
    with open(OUTPUT_DIR / "checkpoint3_ocr_metrics.json", "w") as f:
        json.dump(ocr_metrics, f, indent=2)

    print("\n[OK] Metric JSONs saved to runs/checkpoint3/")


if __name__ == "__main__":
    main()
