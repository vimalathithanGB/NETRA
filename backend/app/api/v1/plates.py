"""License plate captures and ANPR registry API routes."""
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.frontend import PlateCaptureResponse
from app.services.traffic_service import get_recent_plate_captures

router = APIRouter(prefix="/plates", tags=["ANPR Plates"])


@router.get("/captures", response_model=List[PlateCaptureResponse])
def list_plate_captures(limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve recent optical number plate recognition captures."""
    return get_recent_plate_captures(db, limit=limit)
