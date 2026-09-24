# NETRA AI Engine to Backend Integration

Connects the AI perception, tracking, Re-ID, trajectory reconstruction, and predictive traffic analytics pipelines to the NETRA FastAPI backend via standard REST endpoints.

---

## 1. Architectural Overview

```text
CAMERA STREAM / TEST VIDEO (test_2.mp4)
      ↓
VEHICLE DETECTION (YOLOv8n-UVH26)
      ↓
SINGLE-CAMERA TRACKING (ByteTrack)
      ↓
LICENSE PLATE DETECTION & ANPR (YOLOv8n + PaddleOCR)
      ↓
FEATURE RE-IDENTIFICATION (OSNet-AIN 512-d L2 embeddings)
      ↓
CROSS-CAMERA MATCHING (Fusion Evidence Scorer)
      ↓
TRAJECTORY RECONSTRUCTION & SEGMENT KINEMATICS
      ↓
TRAFFIC ANALYTICS & ROUTE PREDICTION (Markov Transition Probabilities)
      ↓
AI OUTPUT FILES (runs/ directory JSON checkpoints)
      ↓ HTTP REST (Batch JSON Payloads)
FASTAPI BACKEND (Validation, Ingestion, & Routing)
      ↓ SQLAlchemy ORM
POSTGRESQL 18 + POSTGIS (netra_db)
      ↓ REST / WebSocket (Compatibility Proxy)
NETRA DASHBOARD / REACT FRONTEND
```

---

## 2. Ingestion Endpoints Utilized

| AI Pipeline Stage | Endpoint | Target Database Tables |
| :--- | :--- | :--- |
| Single-Camera Tracks | `POST /api/v1/ingest/tracks` | `vehicle_observations`, `license_plates` |
| Cross-Camera Matches | `POST /api/v1/ingest/matches` | `cross_camera_matches`, `camera_transitions` |
| Global Vehicles | `POST /api/v1/ingest/global-vehicles` | `global_vehicles`, `vehicle_observations` |
| Trajectories & Segments | `POST /api/v1/ingest/trajectories` | `vehicle_trajectories`, `trajectory_segments` |
| Route Predictions | `POST /api/v1/ingest/predictions` | `route_predictions` |
| Traffic Analytics | `POST /api/v1/analytics/ingest` | Backend Analytics Cache / Summary Store |

---

## 3. Configuration

Set the backend destination URL via environment variable:

```bash
# Windows PowerShell
$env:NETRA_BACKEND_URL = "http://127.0.0.1:8000"

# Linux / macOS
export NETRA_BACKEND_URL="http://127.0.0.1:8000"
```

---

## 4. Execution

To trigger the complete ingestion of verified AI outputs:

```bash
# From workspace root
python -m ai_engine.integration.ai_backend_ingestion
```

Optional arguments:
* `--url`: Explicit backend base URL (e.g. `http://127.0.0.1:8000`)
* `--runs-dir`: Custom runs directory path (defaults to `runs/`)

---

## 5. Simulation Artifact Notice

Calculated speeds (500+ km/h) in the initial trajectory test results are known simulation artifacts resulting from temporal slicing of a single test video (`test_2.mp4`) into 3 simulated camera streams. They are preserved verbatim as required by project specifications to maintain algorithmic integrity without introducing fabricated data.
