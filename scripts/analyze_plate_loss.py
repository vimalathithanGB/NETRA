import json
from pathlib import Path
from collections import defaultdict

ocr_path = Path("runs/checkpoint3/checkpoint3_ocr_metrics.json")
assoc_path = Path("runs/checkpoint3/checkpoint3_association_metrics.json")

with open(ocr_path) as f:
    ocr_data = json.load(f)

with open(assoc_path) as f:
    assoc_data = json.load(f)

print("=== OCR METRICS SUMMARY ===")
for k, v in ocr_data["temporal_aggregation_summary"].items():
    print(f"  {k}: {v}")

print("\n=== TOP 10 OCR TEXTS ===")
for item in ocr_data["distinct_texts_top10"]:
    print(f"  '{item['text']}': count={item['count']}, avg_conf={item['avg_confidence']:.3f}, raw={item['raw_samples']}")

print("\n=== REJECTION CAUSES ===")
for cause, count in assoc_data["rejection_root_causes"].items():
    print(f"  {cause}: {count}")

print("\n=== ALTERNATIVE METHODS COMPARISON ===")
for method, m_data in assoc_data["alternative_methods_comparison"].items():
    print(f"  {method:<30}: Associated={m_data['associated']:>2} ({m_data['association_rate_pct']:>5.1f}%), Rejected={m_data['rejected']:>2}, Ambiguous={m_data['ambiguous_cases']:>2}")
