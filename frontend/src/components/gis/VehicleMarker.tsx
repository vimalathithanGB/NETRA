import React, { useMemo } from 'react';
import { Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { VehicleDetection } from '../../services/gisService';

interface VehicleMarkerProps {
  detection: VehicleDetection;
  index: number;
  isLatest?: boolean;
  onSelect?: (detection: VehicleDetection) => void;
}

export const VehicleMarker: React.FC<VehicleMarkerProps> = ({
  detection,
  index,
  isLatest = false,
  onSelect
}) => {
  const customIcon = useMemo(() => {
    const pulseRing = isLatest
      ? `<span style="position: absolute; inset: -5px; border-radius: 9999px; background: rgba(255, 122, 26, 0.45); animation: pulse-subtle 1.8s infinite;"></span>`
      : '';

    const bg = isLatest ? '#ff7a1a' : '#003366';
    const border = '#ffffff';

    const html = `
      <div style="position: relative; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
        ${pulseRing}
        <div style="
          position: relative;
          width: 28px;
          height: 28px;
          border-radius: 9999px;
          background-color: ${bg};
          border: 2px solid ${border};
          box-shadow: 0 3px 8px rgba(0, 30, 60, 0.35);
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          font-weight: 800;
          font-family: var(--font-mono, monospace);
          font-size: 12px;
        ">
          ${index + 1}
        </div>
      </div>
    `;

    return L.divIcon({
      html,
      className: 'gis-marker-pin',
      iconSize: [34, 34],
      iconAnchor: [17, 17],
      popupAnchor: [0, -18]
    });
  }, [index, isLatest]);

  return (
    <Marker
      position={[detection.latitude, detection.longitude]}
      icon={customIcon}
      eventHandlers={{
        click: () => {
          if (onSelect) onSelect(detection);
        }
      }}
    >
      <Popup minWidth={230} maxWidth={290}>
        <div style={{ fontFamily: 'var(--font-heading, "Public Sans", sans-serif)' }}>
          {/* Header Bar */}
          <div style={{
            backgroundColor: isLatest ? '#ff7a1a' : '#001e40',
            color: '#ffffff',
            padding: '8px 12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderTopLeftRadius: '9px',
            borderTopRightRadius: '9px'
          }}>
            <div>
              <span style={{ fontSize: '10px', textTransform: 'uppercase', opacity: 0.85, display: 'block', fontWeight: 600 }}>
                {isLatest ? 'Current Point' : `Waypoint #${index + 1}`}
              </span>
              <strong style={{
                fontFamily: 'var(--font-mono, monospace)',
                fontSize: '13px',
                letterSpacing: '0.04em'
              }}>
                {detection.plate}
              </strong>
            </div>
            <span style={{
              fontSize: '10px',
              fontWeight: 700,
              padding: '2px 6px',
              borderRadius: '4px',
              backgroundColor: 'rgba(255, 255, 255, 0.25)',
              color: '#ffffff'
            }}>
              {detection.cameraId}
            </span>
          </div>

          {/* Body */}
          <div style={{ padding: '10px 12px', backgroundColor: '#ffffff' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '8px' }}>
              <div style={{ backgroundColor: '#f8fafc', padding: '6px 8px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#64748b', display: 'block', fontSize: '10px' }}>Timestamp</span>
                <strong style={{ color: '#003366', fontSize: '12px', fontFamily: 'var(--font-mono, monospace)' }}>
                  {detection.timestamp}
                </strong>
              </div>

              <div style={{ backgroundColor: '#f8fafc', padding: '6px 8px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#64748b', display: 'block', fontSize: '10px' }}>Vehicle Type</span>
                <strong style={{ color: '#003366', fontSize: '12px' }}>
                  {detection.vehicleType}
                </strong>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '8px' }}>
              <div style={{ backgroundColor: '#f0fdf4', padding: '6px 8px', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <span style={{ color: '#166534', display: 'block', fontSize: '10px' }}>ANPR Confidence</span>
                <strong style={{ color: '#15803d', fontSize: '12px', fontFamily: 'var(--font-mono, monospace)' }}>
                  {detection.confidence}%
                </strong>
              </div>

              <div style={{ backgroundColor: '#f8fafc', padding: '6px 8px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#64748b', display: 'block', fontSize: '10px' }}>Direction</span>
                <strong style={{ color: '#003366', fontSize: '12px' }}>
                  {detection.direction || 'Forward'}
                </strong>
              </div>
            </div>

            {detection.cameraName && (
              <div style={{ fontSize: '11px', color: '#475569', borderTop: '1px solid #f1f5f9', paddingTop: '6px', marginTop: '4px' }}>
                <strong>Node:</strong> {detection.cameraName}
              </div>
            )}
          </div>
        </div>
      </Popup>
    </Marker>
  );
};

export default VehicleMarker;
