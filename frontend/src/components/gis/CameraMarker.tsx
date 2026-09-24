import React, { useMemo } from 'react';
import { Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { Camera } from '../../services/gisService';

interface CameraMarkerProps {
  camera: Camera;
  isHighlighted?: boolean;
  isLatest?: boolean;
  detectionBadge?: string;
  sequenceIndex?: number;
  onSelect?: (camera: Camera) => void;
}

export const CameraMarker: React.FC<CameraMarkerProps> = ({
  camera,
  isHighlighted = false,
  isLatest = false,
  detectionBadge,
  sequenceIndex,
  onSelect
}) => {
  // Determine color palette based on camera status
  const statusConfig = useMemo(() => {
    switch (camera.status) {
      case 'online':
        return {
          bg: '#10b981',
          border: '#059669',
          glow: 'rgba(16, 185, 129, 0.45)',
          label: 'Online',
          pulse: true
        };
      case 'warning':
        return {
          bg: '#f59e0b',
          border: '#d97706',
          glow: 'rgba(245, 158, 11, 0.45)',
          label: 'Degraded',
          pulse: false
        };
      case 'offline':
      default:
        return {
          bg: '#ef4444',
          border: '#dc2626',
          glow: 'rgba(239, 68, 68, 0.45)',
          label: 'Offline',
          pulse: false
        };
    }
  }, [camera.status]);

  // Create custom DivIcon for Leaflet
  const customIcon = useMemo(() => {
    const pulseRing = isLatest
      ? `<span style="position: absolute; inset: -8px; border-radius: 9999px; background: rgba(255, 122, 26, 0.55); animation: pulse 1.5s infinite;"></span>`
      : isHighlighted
      ? `<span style="position: absolute; inset: -8px; border-radius: 9999px; background: rgba(14, 165, 233, 0.55); animation: pulse 1.5s infinite;"></span>`
      : statusConfig.pulse
      ? `<span style="position: absolute; inset: -4px; border-radius: 9999px; background: ${statusConfig.glow}; animation: pulse-subtle 2s infinite;"></span>`
      : '';

    const badgeText = detectionBadge || (sequenceIndex !== undefined ? `${sequenceIndex + 1}` : '');
    const badgeBg = isLatest ? '#ff7a1a' : '#0284c7';
    const badgeHtml = badgeText
      ? `<span style="position: absolute; top: -7px; right: -7px; background: ${badgeBg}; color: #ffffff; border: 2px solid #ffffff; font-size: 10px; font-weight: 800; border-radius: 9999px; min-width: 18px; height: 18px; padding: 0 3px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;">${badgeText}</span>`
      : '';

    const pinBg = isLatest ? '#ff7a1a' : (isHighlighted ? '#0284c7' : statusConfig.bg);
    const pinBorder = (isLatest || isHighlighted) ? '3px solid #ffffff' : '2px solid #ffffff';
    const pinShadow = isLatest
      ? '0 0 16px rgba(255, 122, 26, 0.95), 0 2px 8px rgba(0,0,0,0.4)'
      : isHighlighted
      ? '0 0 14px rgba(2, 132, 199, 0.9), 0 2px 8px rgba(0,0,0,0.4)'
      : '0 2px 6px rgba(0,0,0,0.3)';

    const html = `
      <div style="position: relative; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
        ${pulseRing}
        ${badgeHtml}
        <div style="
          position: relative;
          width: ${(isHighlighted || isLatest) ? '28px' : '26px'};
          height: ${(isHighlighted || isLatest) ? '28px' : '26px'};
          border-radius: 9999px;
          background-color: ${pinBg};
          border: ${pinBorder};
          box-shadow: ${pinShadow};
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          font-weight: 800;
          font-size: 11px;
          transition: transform 0.15s ease;
        ">
          <svg style="width: 14px; height: 14px;" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
          </svg>
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
  }, [statusConfig, isHighlighted, isLatest, sequenceIndex, detectionBadge]);

  const roadDisplay = camera.roadName || camera.road_name || 'N/A';
  const cityDisplay = camera.city || 'Coimbatore';
  const speedLimit = camera.speedLimitKmh || camera.speed_limit_kmh;

  return (
    <Marker
      position={[camera.latitude, camera.longitude]}
      icon={customIcon}
      eventHandlers={{
        click: () => {
          if (onSelect) onSelect(camera);
        }
      }}
    >
      <Popup minWidth={240} maxWidth={300}>
        <div style={{ fontFamily: 'var(--font-heading, "Public Sans", sans-serif)' }}>
          {/* Header Bar */}
          <div style={{
            backgroundColor: isLatest ? '#9a3412' : (isHighlighted ? '#002855' : '#001e40'),
            color: '#ffffff',
            padding: '8px 12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderTopLeftRadius: '9px',
            borderTopRightRadius: '9px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{
                fontFamily: 'var(--font-mono, monospace)',
                fontWeight: 700,
                fontSize: '12px',
                letterSpacing: '0.04em'
              }}>
                {camera.id}
              </span>
              {isLatest ? (
                <span style={{
                  fontSize: '9px',
                  fontWeight: 800,
                  padding: '1px 5px',
                  borderRadius: '4px',
                  backgroundColor: '#ff7a1a',
                  color: '#ffffff',
                  textTransform: 'uppercase'
                }}>
                  Last Detected
                </span>
              ) : isHighlighted ? (
                <span style={{
                  fontSize: '9px',
                  fontWeight: 800,
                  padding: '1px 5px',
                  borderRadius: '4px',
                  backgroundColor: '#0284c7',
                  color: '#ffffff',
                  textTransform: 'uppercase'
                }}>
                  Tracked Node
                </span>
              ) : null}
            </div>
            <span style={{
              fontSize: '10px',
              fontWeight: 700,
              padding: '2px 7px',
              borderRadius: '9999px',
              backgroundColor: statusConfig.bg,
              color: '#ffffff',
              textTransform: 'uppercase'
            }}>
              {statusConfig.label}
            </span>
          </div>

          {/* Body */}
          <div style={{ padding: '10px 12px', backgroundColor: '#ffffff' }}>
            <div style={{ fontSize: '13px', fontWeight: 800, color: '#071c36', marginBottom: '4px' }}>
              {camera.name}
            </div>

            <div style={{ fontSize: '11px', color: '#475569', marginBottom: '8px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <div><strong>Road:</strong> {roadDisplay}</div>
              <div><strong>City:</strong> {cityDisplay}</div>
              {speedLimit && <div><strong>Speed Limit:</strong> {speedLimit} km/h</div>}
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '6px',
              padding: '8px',
              backgroundColor: '#f8fafc',
              borderRadius: '6px',
              border: '1px solid #e2e8f0',
              fontSize: '11px',
              marginBottom: '8px'
            }}>
              <div>
                <span style={{ color: '#64748b', display: 'block', fontSize: '10px' }}>Last Detection</span>
                <strong style={{ color: '#003366', fontFamily: 'var(--font-mono, monospace)' }}>
                  {camera.lastDetection || 'N/A'}
                </strong>
              </div>
              <div>
                <span style={{ color: '#64748b', display: 'block', fontSize: '10px' }}>Vehicles Detected</span>
                <strong style={{ color: '#003366', fontFamily: 'var(--font-mono, monospace)' }}>
                  {camera.vehicleCount !== undefined ? camera.vehicleCount : '0'}
                </strong>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '10px', color: '#64748b', borderTop: '1px solid #f1f5f9', paddingTop: '6px' }}>
              <span>GPS: {camera.latitude.toFixed(5)}°N, {camera.longitude.toFixed(5)}°E</span>
              <span>Status: <strong style={{ color: statusConfig.border }}>{camera.status}</strong></span>
            </div>
          </div>
        </div>
      </Popup>
    </Marker>
  );
};

export default CameraMarker;

