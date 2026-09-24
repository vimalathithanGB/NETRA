"""Traffic status, ANPR captures, and alert events service."""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.entities import LicensePlate, VehicleObservation, Camera, Alert, TrafficEvent
from app.schemas.frontend import (
    TrafficZone,
    TrafficIncident,
    TrafficIncidentCreate,
    TrafficEventsResponse,
    PlateCaptureResponse,
    AlertResponse,
    AlertCreate,
)

# Standard Delhi-NCR Smart Grid Traffic Zones
DEFAULT_TRAFFIC_ZONES = [
    TrafficZone(
        id="north-zone",
        name="North Zone",
        status="normal",
        statusText="Normal Flow (18 km/h avg)",
        color="#138808",
        center=(28.6448, 77.2167),
        radiusMeters=1400,
        density="Low",
        avgSpeed="42 km/h",
    ),
    TrafficZone(
        id="central-hub",
        name="Central Hub",
        status="congestion",
        statusText="Congestion Warning (8 km/h)",
        color="#f59e0b",
        center=(28.6139, 77.2090),
        radiusMeters=1600,
        density="High",
        avgSpeed="12 km/h",
    ),
    TrafficZone(
        id="east-corridor",
        name="East Corridor",
        status="normal",
        statusText="Smooth (45 km/h avg)",
        color="#138808",
        center=(28.6280, 77.2789),
        radiusMeters=1800,
        density="Low",
        avgSpeed="48 km/h",
    ),
    TrafficZone(
        id="port-access",
        name="Port Access & Logistics",
        status="incident",
        statusText="Incident Reported (Lane 2 Blocked)",
        color="#ba1a1a",
        center=(28.5355, 77.2410),
        radiusMeters=1300,
        density="Restricted",
        avgSpeed="6 km/h",
    ),
]

# Baseline Incident Overlay Markers
DEFAULT_INCIDENTS = [
    TrafficIncident(
        id="INC-7802",
        title="Lane 2 Bottleneck & Breakdown",
        severity="high",
        latitude=28.5355,
        longitude=77.2410,
        timestamp="14:05 IST",
        corridor="Port Access Arterial",
        description="Commercial carrier disabled on outbound corridor. Patrol dispatched.",
    ),
    TrafficIncident(
        id="INC-7805",
        title="High Density Crawl at Flyover Entry",
        severity="medium",
        latitude=28.5672,
        longitude=77.2100,
        timestamp="14:28 IST",
        corridor="Ring Road & Aurobindo Marg",
        description="Vehicle density spike above 85% threshold.",
    ),
]

DEFAULT_ALERTS = [
    AlertResponse(
        id="ALT-901",
        type="Overspeeding (>110 km/h)",
        plate="DL 04 C 8891",
        location="DND Flyway Entry North",
        severity="Critical",
        time="14:23 IST",
        status="Pending Review",
    ),
    AlertResponse(
        id="ALT-902",
        type="Wrong Way Arterial Ingress",
        plate="UP 16 F 3301",
        location="Ring Road AIIMS Underpass",
        severity="Critical",
        time="14:20 IST",
        status="Pending Review",
    ),
    AlertResponse(
        id="ALT-903",
        type="Stalled Heavy Freight - Lane Block",
        plate="HR 26 DK 5092",
        location="Port Access Arterial",
        severity="High",
        time="14:18 IST",
        status="Under Investigation",
    ),
    AlertResponse(
        id="ALT-904",
        type="Non-Standard Obscured Number Plate",
        plate="BR 01 AB 1234",
        location="Vikas Marg ITO Junction",
        severity="Medium",
        time="14:12 IST",
        status="Pending Review",
    ),
    AlertResponse(
        id="ALT-905",
        type="Red Light Jump Violation",
        plate="KA 01 M 9912",
        location="Connaught Place Outer Circle",
        severity="Medium",
        time="14:02 IST",
        status="e-Challan Queued",
    ),
]


def seed_default_alerts(db: Session) -> int:
    """Seed baseline enforcement alerts into PostgreSQL if not present."""
    count = 0
    for a in DEFAULT_ALERTS:
        if not db.get(Alert, a.id):
            record = Alert(
                id=a.id,
                alert_type=a.type,
                severity=a.severity,
                plate_text=a.plate,
                location_text=a.location,
                timestamp_text=a.time,
                status=a.status,
            )
            db.add(record)
            count += 1
    if count > 0:
        db.commit()
    return count


def seed_default_traffic_events(db: Session) -> int:
    """Seed baseline incident overlay events into PostgreSQL if not present."""
    count = 0
    for inc in DEFAULT_INCIDENTS:
        if not db.get(TrafficEvent, inc.id):
            wkt_geom = f"SRID=4326;POINT({inc.longitude} {inc.latitude})"
            event = TrafficEvent(
                id=inc.id,
                event_type="incident",
                title=inc.title,
                severity=inc.severity,
                corridor=inc.corridor,
                latitude=inc.latitude,
                longitude=inc.longitude,
                location_geom=wkt_geom,
                description=inc.description,
                timestamp_text=inc.timestamp,
                status="active",
            )
            db.add(event)
            count += 1
    if count > 0:
        db.commit()
    return count


def get_traffic_events(db: Optional[Session] = None) -> TrafficEventsResponse:
    """Return active traffic zones and incident telemetry from PostgreSQL or defaults."""
    if db is not None:
        events = db.execute(select(TrafficEvent).order_by(TrafficEvent.id)).scalars().all()
        if not events:
            seed_default_traffic_events(db)
            events = db.execute(select(TrafficEvent).order_by(TrafficEvent.id)).scalars().all()

        incidents = [
            TrafficIncident(
                id=e.id,
                title=e.title,
                severity=e.severity,
                latitude=e.latitude if e.latitude is not None else 28.5355,
                longitude=e.longitude if e.longitude is not None else 77.2410,
                timestamp=e.timestamp_text or "14:05 IST",
                corridor=e.corridor or "Arterial Corridor",
                description=e.description or "",
            )
            for e in events
        ]
        return TrafficEventsResponse(
            zones=DEFAULT_TRAFFIC_ZONES,
            incidents=incidents,
        )

    return TrafficEventsResponse(
        zones=DEFAULT_TRAFFIC_ZONES,
        incidents=DEFAULT_INCIDENTS,
    )


def get_recent_plate_captures(db: Session, limit: int = 100) -> List[PlateCaptureResponse]:
    """Return recent ANPR OCR captures from the database."""
    obs_list = db.execute(
        select(VehicleObservation)
        .where(VehicleObservation.plate_text.isnot(None))
        .order_by(VehicleObservation.id.desc())
        .limit(limit)
    ).scalars().all()

    if not obs_list:
        # Fallback to LicensePlate index
        plates = db.execute(
            select(LicensePlate)
            .order_by(LicensePlate.id.desc())
            .limit(limit)
        ).scalars().all()
        return [
            PlateCaptureResponse(
                id=p.id,
                plate=p.plate_text,
                type=p.vehicle_class or "Vehicle",
                camId=p.last_camera_id or "CAM-001",
                timestamp=p.last_seen_timestamp.strftime("%H:%M:%S") if p.last_seen_timestamp else "14:22:00",
                confidence=f"{p.confidence:.1f}%" if p.confidence else "98.5%",
                status="Verified" if p.plate_status == "stable" else "Review",
            )
            for p in plates
        ]

    return [
        PlateCaptureResponse(
            id=o.id,
            plate=o.plate_text or "UNKNOWN",
            type=o.vehicle_class,
            camId=o.camera_id,
            timestamp=o.created_at.strftime("%H:%M:%S") if o.created_at else "14:22:00",
            confidence=f"{(o.best_ocr_confidence or o.plate_confidence or 98.0):.1f}%",
            status="Verified" if o.plate_status == "stable" else "Review",
        )
        for o in obs_list
    ]


def get_active_alerts(db: Session) -> List[AlertResponse]:
    """Return active traffic violations and critical enforcement alerts from PostgreSQL."""
    alerts = db.execute(select(Alert).order_by(Alert.id)).scalars().all()
    if not alerts:
        seed_default_alerts(db)
        alerts = db.execute(select(Alert).order_by(Alert.id)).scalars().all()

    return [
        AlertResponse(
            id=a.id,
            type=a.alert_type,
            plate=a.plate_text or "UNKNOWN",
            location=a.location_text or "Smart Surveillance Corridor",
            severity=a.severity,
            time=a.timestamp_text or (a.created_at.strftime("%H:%M IST") if a.created_at else "14:20 IST"),
            status=a.status,
        )
        for a in alerts
    ]


def create_alert(db: Session, alert_in: AlertCreate) -> AlertResponse:
    """Create a new persistent alert in PostgreSQL."""
    alert_id = alert_in.id or f"ALT-{int(db.scalar(select(Alert.id).count()) or 0) + 901}"
    record = Alert(
        id=alert_id,
        alert_type=alert_in.alert_type,
        severity=alert_in.severity or "Medium",
        message=alert_in.message,
        camera_id=alert_in.camera_id,
        global_vehicle_id=alert_in.global_vehicle_id,
        plate_text=alert_in.plate_text,
        location_text=alert_in.location_text,
        timestamp_text=alert_in.timestamp_text,
        status=alert_in.status or "Pending Review",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return AlertResponse(
        id=record.id,
        type=record.alert_type,
        plate=record.plate_text or "UNKNOWN",
        location=record.location_text or "Corridor Node",
        severity=record.severity,
        time=record.timestamp_text or "Just now",
        status=record.status,
    )


def create_traffic_incident(db: Session, inc_in: TrafficIncidentCreate) -> TrafficIncident:
    """Create a new persistent traffic incident in PostgreSQL."""
    inc_id = inc_in.id or f"INC-{int(db.scalar(select(TrafficEvent.id).count()) or 0) + 7801}"
    wkt_geom = f"SRID=4326;POINT({inc_in.longitude or 77.2100} {inc_in.latitude or 28.5672})"
    record = TrafficEvent(
        id=inc_id,
        event_type=inc_in.event_type or "incident",
        title=inc_in.title,
        severity=inc_in.severity or "medium",
        camera_id=inc_in.camera_id,
        corridor=inc_in.corridor,
        latitude=inc_in.latitude,
        longitude=inc_in.longitude,
        location_geom=wkt_geom,
        description=inc_in.description,
        timestamp_text=inc_in.timestamp_text or "Now",
        status=inc_in.status or "active",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return TrafficIncident(
        id=record.id,
        title=record.title,
        severity=record.severity,
        latitude=record.latitude or 28.5672,
        longitude=record.longitude or 77.2100,
        timestamp=record.timestamp_text or "Now",
        corridor=record.corridor or "Corridor",
        description=record.description or "",
    )

