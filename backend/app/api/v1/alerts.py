from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.frontend import AlertResponse, AlertCreate
from app.services.traffic_service import get_active_alerts, create_alert

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=List[AlertResponse])
def list_alerts(db: Session = Depends(get_db)):
    """Retrieve active critical violations and enforcement alerts from PostgreSQL."""
    return get_active_alerts(db)


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
def register_alert(alert_in: AlertCreate, db: Session = Depends(get_db)):
    """Register a persistent critical traffic violation or e-Challan enforcement alert."""
    return create_alert(db, alert_in)
