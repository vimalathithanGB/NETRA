"""Camera surveillance node API routes."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.frontend import CameraResponse, CameraCreate
from app.services.camera_service import (
    get_all_cameras,
    get_camera,
    create_or_update_camera,
    _format_camera_response,
)

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.get("", response_model=List[CameraResponse])
def list_cameras(db: Session = Depends(get_db)):
    """Retrieve all operational surveillance camera nodes in the grid."""
    return get_all_cameras(db)


@router.get("/{camera_id}", response_model=CameraResponse)
def retrieve_camera(camera_id: str, db: Session = Depends(get_db)):
    """Retrieve specific camera node by ID."""
    cam = get_camera(db, camera_id)
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera {camera_id} not found")
    return _format_camera_response(cam)


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def register_camera(camera_in: CameraCreate, db: Session = Depends(get_db)):
    """Register or update a camera node with PostGIS coordinates."""
    return create_or_update_camera(db, camera_in)
