# NETRA Backend

Networked Engine for Traffic Recognition & Analytics (NETRA) — Backend Service.

---

## 1. Architecture

```text
CAMERA / VIDEO STREAM
         ↓
     AI ENGINE (Detection, OCR, Re-ID, Tracking)
         ↓
  FASTAPI BACKEND (Validation, Ingestion, Business Logic, APIs)
         ↓
POSTGRESQL + POSTGIS (Spatio-Temporal Datastore, Indices, Relational Telemetry)
         ↓
NETRA FRONTEND / GUI (Dashboard, Multi-Camera Trajectory Map, ANPR, Alerts)
```

---

## 2. Requirements

- **Python**: 3.11+
- **PostgreSQL**: 14+ (with PostGIS extension)
- **PostGIS**: 3.0+
- **Git**: 2.x+

---

## 3. Setup & Installation

### 3.1 Clone & Navigate
```bash
cd backend
```

### 3.2 Create & Activate Python Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Database Setup (PostgreSQL + PostGIS)

Ensure PostgreSQL is running locally or remotely, then create the `netra_db` database:

```sql
-- 1. Create the database
CREATE DATABASE netra_db;

-- 2. Connect to the database
\c netra_db

-- 3. Enable PostGIS extension for spatial data & queries
CREATE EXTENSION IF NOT EXISTS postgis;

-- 4. Enable Trigram extension for fast fuzzy plate search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 5. Verify PostGIS installation
SELECT PostGIS_Version();
```

---

## 5. Environment Variables Configuration

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

Key variables in `.env`:
```env
APP_NAME="NETRA Traffic Intelligence Backend"
APP_ENV="development"
DEBUG=true

# Database connection string
DATABASE_URL="postgresql://postgres:YOUR_PASSWORD@localhost:5432/netra_db"

# Security key for JWT tokens
SECRET_KEY="YOUR_SUPER_SECRET_KEY_HERE"
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Routing prefix
API_V1_PREFIX="/api/v1"

# Frontend CORS origins
CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

---

## 6. Running the Backend Server

Start the development server with live reload:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once started:
- **API Base URL:** `http://127.0.0.1:8000`
- **Health Check:** `http://127.0.0.1:8000/health`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **ReDoc Documentation:** `http://127.0.0.1:8000/redoc`

---

## 7. Running Automated Tests

Run the test suite using `pytest`:

```bash
pytest
```

---

## 8. Database Migrations (Alembic)

Alembic is pre-configured with `app.core.config.settings.DATABASE_URL` and `app.db.base.Base.metadata`.

```bash
# Check current migration head
alembic heads

# Generate a new migration after adding SQLAlchemy models
alembic revision --autogenerate -m "create initial tables"

# Apply migrations
alembic upgrade head
```

---

## 9. Project Directory Structure

```text
backend/
├── app/
│   ├── __init__.py           # Package marker
│   ├── main.py               # FastAPI application setup, CORS, and health check
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── router.py     # Master v1 API router & AI ingestion placeholder
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Pydantic Settings configuration loader
│   │   └── security.py       # Bcrypt hashing and python-jose JWT token utilities
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py        # SQLAlchemy engine, sessionmaker, and get_db dependency
│   │   └── base.py           # DeclarativeBase class
│   ├── models/
│   │   └── __init__.py       # SQLAlchemy model registry (provisional entities documented)
│   ├── schemas/
│   │   └── __init__.py       # Pydantic data validation schemas
│   ├── services/
│   │   └── __init__.py       # Business logic and service layer
│   └── utils/
│       └── __init__.py       # Utility functions and helper tools
├── tests/
│   ├── __init__.py
│   ├── test_health.py        # Tests for /health, /api/v1/status, and /docs
│   └── test_security.py      # Tests for password hashing & JWT token verification
├── alembic/
│   ├── env.py                # Migration runner wired to app settings & metadata
│   ├── script.py.mako        # Migration template
│   └── versions/             # Migration version scripts
├── .env                      # Local environment configuration (ignored by git)
├── .env.example              # Template environment configuration
├── .gitignore                # Git ignore rules
├── alembic.ini               # Alembic configuration file
├── requirements.txt          # Frozen project dependencies
└── README.md                 # Project documentation
```

---

## 10. Current Status

> **Backend foundation created.**
> - Final AI → Backend contract is pending output from the AI Engine developer.
> - Final database schema is pending the comparison between real AI JSON output and frontend GUI requirements.
> - The application is fully scaffolded, tested, and ready for integration.
