import React, { useMemo } from 'react';
import { Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { TrafficIncident } from '../../services/gisService';

interface IncidentMarkerProps {
  incident: TrafficIncident;
  onSelect?: (incident: TrafficIncident) => void;
}

export const IncidentMarker: React.FC<IncidentMarkerProps> = ({ incident, onSelect }) => {
  const isCritical = incident.severity === 'critical' || incident.severity === 'high';
  const color = isCritical ? '#dc2626' : '#ea580c';
  const glow = isCritical ? 'rgba(220, 38, 38, 0.45)' : 'rgba(234, 88, 12, 0.45)';

  const customIcon = useMemo(() => {
    const html = `
      <div style="position: relative; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
        <span style="position: absolute; inset: -4px; border-radius: 9999px; background: ${glow}; animation: pulse-subtle 2s infinite;"></span>
        <div style="
          position: relative;
          width: 24px;
          height: 24px;
          border-radius: 6px;
          background-color: ${color};
          border: 2px solid #ffffff;
          box-shadow: 0 2px 6px rgba(0,0,0,0.35);
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
        ">
          <svg style="width: 13px; height: 13px;" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
          </svg>
        </div>
      </div>
    `;

    return L.divIcon({
      html,
      className: 'gis-incident-pin',
      iconSize: [30, 30],
      iconAnchor: [15, 15],
      popupAnchor: [0, -16]
    });
  }, [color, glow]);

  if (!incident.latitude || !incident.longitude) return null;

  return (
    <Marker
      position={[incident.latitude, incident.longitude]}
      icon={customIcon}
      eventHandlers={{
        click: () => {
          if (onSelect) onSelect(incident);
        }
      }}
    >
      <Popup minWidth={220} maxWidth={280}>
        <div style={{ fontFamily: 'var(--font-heading, "Public Sans", sans-serif)' }}>
          <div style={{
            backgroundColor: color,
            color: '#ffffff',
            padding: '7px 10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderTopLeftRadius: '7px',
            borderTopRightRadius: '7px',
            fontSize: '11px',
            fontWeight: 800,
            textTransform: 'uppercase'
          }}>
            <span>Incident Alert</span>
            <span style={{ backgroundColor: 'rgba(255,255,255,0.25)', padding: '1px 5px', borderRadius: '3px' }}>
              {incident.severity}
            </span>
          </div>
          <div style={{ padding: '8px 10px', backgroundColor: '#ffffff', fontSize: '11px', color: '#1e293b' }}>
            <div style={{ fontWeight: 700, fontSize: '12px', marginBottom: '4px', color: '#0f172a' }}>
              {incident.title}
            </div>
            <div style={{ color: '#64748b', marginBottom: '4px' }}>
              <strong>Corridor:</strong> {incident.corridor}
            </div>
            {incident.timestamp && (
              <div style={{ color: '#64748b', marginBottom: '4px' }}>
                <strong>Reported:</strong> {incident.timestamp}
              </div>
            )}
            {incident.description && (
              <div style={{ color: '#475569', fontSize: '10.5px', marginTop: '4px', borderTop: '1px dashed #e2e8f0', paddingTop: '4px' }}>
                {incident.description}
              </div>
            )}
          </div>
        </div>
      </Popup>
    </Marker>
  );
};
