"""
NETRA — Real Multi-Camera Common Vehicle Image Test
Script: scripts/inspect_real_multicamera_input.py
Step 1: Input Inventory & Verification
"""
import os
import sys
import json
import hashlib
from pathlib import Path
from PIL import Image, ExifTags

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "images"

FOLDERS = {
    "camera 1": "CAM_01",
    "camera 2": "CAM_02",
    "camera 3": "CAM_03",
    "camera 4": "CAM_04"
}

def main():
    inventory = {
        "summary": {
            "total_folders": len(FOLDERS),
            "total_images": 0,
            "corrupted_images": 0,
            "duplicate_images_count": 0
        },
        "camera_folders": {},
        "duplicate_groups": [],
        "images": []
    }

    hash_map = {}
    total_images = 0
    corrupted_count = 0

    for folder_name, cam_id in FOLDERS.items():
        folder_path = BASE_DIR / folder_name
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist")
            continue
        
        files = sorted(list(folder_path.iterdir()), key=lambda p: p.name)
        cam_info = {
            "folder_name": folder_name,
            "camera_id": cam_id,
            "image_count": len(files),
            "files": []
        }
        
        for idx, file_path in enumerate(files):
            total_images += 1
            img_info = {
                "camera_id": cam_id,
                "folder_name": folder_name,
                "filename": file_path.name,
                "relative_path": str(file_path.relative_to(PROJECT_ROOT).as_posix()),
                "file_size_bytes": file_path.stat().st_size,
                "format": None,
                "dimensions": None,
                "readable": False,
                "error": None,
                "md5_hash": None,
                "exif_timestamps_available": False,
                "exif_data": {}
            }
            
            # Compute MD5
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                    file_hash = hashlib.md5(content).hexdigest()
                    img_info["md5_hash"] = file_hash
                    if file_hash in hash_map:
                        hash_map[file_hash].append(str(file_path.relative_to(PROJECT_ROOT).as_posix()))
                    else:
                        hash_map[file_hash] = [str(file_path.relative_to(PROJECT_ROOT).as_posix())]
            except Exception as e:
                img_info["error"] = f"Hash error: {str(e)}"
                
            # Open and inspect with PIL
            try:
                with Image.open(file_path) as img:
                    img_info["format"] = img.format
                    img_info["dimensions"] = {"width": img.width, "height": img.height}
                    img_info["readable"] = True
                    
                    # Check EXIF
                    exif_data = img.getexif()
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            if "Date" in tag_name or "Time" in tag_name:
                                img_info["exif_timestamps_available"] = True
                                img_info["exif_data"][tag_name] = str(value)
            except Exception as e:
                img_info["error"] = f"Image read error: {str(e)}"
                corrupted_count += 1
                
            cam_info["files"].append(img_info["filename"])
            inventory["images"].append(img_info)
            
        inventory["camera_folders"][cam_id] = cam_info

    # Detect duplicates
    dup_count = 0
    for h, path_list in hash_map.items():
        if len(path_list) > 1:
            inventory["duplicate_groups"].append({
                "md5": h,
                "occurrences": path_list
            })
            dup_count += (len(path_list) - 1)

    inventory["summary"]["total_images"] = total_images
    inventory["summary"]["corrupted_images"] = corrupted_count
    inventory["summary"]["duplicate_images_count"] = dup_count

    out_dir = PROJECT_ROOT / "runs" / "real_multicamera_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "input_inventory.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print("==================================================")
    print("NETRA REAL MULTI-CAMERA INPUT INVENTORY")
    print("==================================================")
    print(f"Total Images: {total_images}")
    print(f"Corrupted Images: {corrupted_count}")
    print(f"Duplicate Images Count: {dup_count}")
    for cam_id, info in inventory["camera_folders"].items():
        print(f"  {cam_id} ({info['folder_name']}): {info['image_count']} images")
        for fn in info["files"]:
            item = next(im for im in inventory["images"] if im["camera_id"] == cam_id and im["filename"] == fn)
            dims = f"{item['dimensions']['width']}x{item['dimensions']['height']}" if item["dimensions"] else "N/A"
            exif_status = "EXIF Date: Yes" if item["exif_timestamps_available"] else "EXIF Date: None"
            print(f"    - {fn} ({item['format']}, {dims}, {item['file_size_bytes']} bytes, {exif_status})")
    print(f"Saved inventory to: {out_file}")

if __name__ == "__main__":
    main()
