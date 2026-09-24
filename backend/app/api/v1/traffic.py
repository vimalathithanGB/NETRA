from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.frontend import TrafficEventsResponse, TrafficIncident, TrafficIncidentCreate
from app.services.traffic_service import get_traffic_events, create_traffic_incident

router = APIRouter(prefix="/traffic", tags=["Traffic"])


@router.get("/events", response_model=TrafficEventsResponse)
def list_traffic_events(db: Session = Depends(get_db)):
    """Retrieve active traffic zones, density corridors, and incident alerts from PostgreSQL."""
    return get_traffic_events(db)


@router.post("/events/incidents", response_model=TrafficIncident, status_code=status.HTTP_201_CREATED)
def register_traffic_incident(inc_in: TrafficIncidentCreate, db: Session = Depends(get_db)):
    """Register an active incident alert pin in the traffic network."""
    return create_traffic_incident(db, inc_in)

