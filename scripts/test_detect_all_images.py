"""
NETRA — Inspect Detections on All 10 Real Images
"""
import os
import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "images"

FOLDERS = {
    "camera 1": "CAM_01",
    "camera 2": "CAM_02",
    "camera 3": "CAM_03",
    "camera 4": "CAM_04"
}

def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    v_model = YOLO("runs/vehicle_detection/vehicle_yolov8n_uvh26/weights/best.pt")
    p_model = YOLO("runs/detect/runs/plate_detection/vehicle_plate_yolov8n_50ep/weights/best.pt")

    for folder_name, cam_id in FOLDERS.items():
        p = BASE_DIR / folder_name
        files = sorted(list(p.iterdir()), key=lambda x: x.name)
        print(f"\n==================== {cam_id} ({folder_name}) ====================")
        for img_path in files:
            img = cv2.imread(str(img_path))
            v_res = v_model.predict(img, conf=0.35, imgsz=640, device=dev, verbose=False)[0]
            p_res = p_model.predict(img, conf=0.30, imgsz=1280, device=dev, verbose=False)[0]
            print(f"Image {img_path.name}: {len(v_res.boxes)} vehicles, {len(p_res.boxes)} plates")
            for i, box in enumerate(v_res.boxes):
                cls_name = v_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                print(f"   [V{i}] {cls_name:12s} conf={conf:.3f} bbox={xyxy}")
            for j, pbox in enumerate(p_res.boxes):
                pconf = float(pbox.conf[0])
                pxyxy = pbox.xyxy[0].cpu().numpy().astype(int).tolist()
                print(f"   [P{j}] plate conf={pconf:.3f} bbox={pxyxy}")

if __name__ == "__main__":
    main()
