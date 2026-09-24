"""AI Engine output ingestion API routes."""
import logging
from typing import List, Union
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.ai_contracts import (
    CameraTracksIngest,
    CrossCameraMatchInput,
    GlobalVehicleInput,
    TrajectoryInput,
    RoutePredictionInput,
    UnifiedAIIngestPayload,
    IngestResponse,
)
from app.services.ingest_service import (
    ingest_tracks,
    ingest_matches,
    ingest_global_vehicles,
    ingest_trajectories,
    ingest_predictions,
    ingest_unified,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["AI Ingestion"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_all(payload: UnifiedAIIngestPayload, db: Session = Depends(get_db)):
    """Unified ingestion endpoint accepting any confirmed AI outputs."""
    try:
        counts = ingest_unified(db, payload)
        return IngestResponse(
            success=True,
            message="AI Engine data ingested successfully",
            counts=counts,
        )
    except Exception as e:
        logger.error(f"Failed to ingest unified AI payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest AI payload: {str(e)}",
        )


@router.post("/tracks", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_camera_tracks(payload: CameraTracksIngest, db: Session = Depends(get_db)):
    """Ingest confirmed Vehicle Track JSON outputs for a given camera."""
    try:
        count = ingest_tracks(db, payload.camera_id, payload.tracks)
        return IngestResponse(
            success=True,
            message=f"Ingested {count} tracks for camera {payload.camera_id}",
            counts={"tracks": count},
        )
    except Exception as e:
        logger.error(f"Failed to ingest camera tracks: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest camera tracks: {str(e)}",
        )


@router.post("/matches", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_cross_camera_matches(
    matches: Union[CrossCameraMatchInput, List[CrossCameraMatchInput]],
    db: Session = Depends(get_db),
):
    """Ingest confirmed Cross-Camera Match JSON outputs."""
    try:
        match_list = [matches] if isinstance(matches, CrossCameraMatchInput) else matches
        count = ingest_matches(db, match_list)
        return IngestResponse(
            success=True,
            message=f"Ingested {count} cross-camera matches",
            counts={"matches": count},
        )
    except Exception as e:
        logger.error(f"Failed to ingest cross-camera matches: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest cross-camera matches: {str(e)}",
        )


@router.post("/global-vehicles", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_vehicles(
    vehicles: Union[GlobalVehicleInput, List[GlobalVehicleInput]],
    db: Session = Depends(get_db),
):
    """Ingest confirmed Global Vehicle JSON outputs."""
    try:
        vehicle_list = [vehicles] if isinstance(vehicles, GlobalVehicleInput) else vehicles
        count = ingest_global_vehicles(db, vehicle_list)
        return IngestResponse(
            success=True,
            message=f"Ingested {count} global vehicle records",
            counts={"global_vehicles": count},
        )
    except Exception as e:
        logger.error(f"Failed to ingest global vehicles: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest global vehicles: {str(e)}",
        )


@router.post("/trajectories", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_vehicle_trajectories(
    trajectories: Union[TrajectoryInput, List[TrajectoryInput]],
    db: Session = Depends(get_db),
):
    """Ingest confirmed Trajectory JSON outputs with segments."""
    try:
        traj_list = [trajectories] if isinstance(trajectories, TrajectoryInput) else trajectories
        count = ingest_trajectories(db, traj_list)
        return IngestResponse(
            success=True,
            message=f"Ingested {count} trajectories",
            counts={"trajectories": count},
        )
    except Exception as e:
        logger.error(f"Failed to ingest vehicle trajectories: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest vehicle trajectories: {str(e)}",
        )


@router.post("/predictions", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_route_predictions(
    predictions: Union[RoutePredictionInput, List[RoutePredictionInput]],
    db: Session = Depends(get_db),
):
    """Ingest confirmed Route Prediction JSON outputs."""
    try:
        pred_list = [predictions] if isinstance(predictions, RoutePredictionInput) else predictions
        count = ingest_predictions(db, pred_list)
        return IngestResponse(
            success=True,
            message=f"Ingested {count} route predictions",
            counts={"predictions": count},
        )
    except Exception as e:
        logger.error(f"Failed to ingest route predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest route predictions: {str(e)}",
        )
