import React from 'react';

interface MapLegendProps {
  position?: 'topright' | 'topleft' | 'bottomright' | 'bottomleft';
  showLayers?: boolean;
}

export const MapLegend: React.FC<MapLegendProps> = ({ position = 'bottomleft' }) => {
  const positionStyles: Record<string, React.CSSProperties> = {
    topright: { top: '12px', right: '12px' },
    topleft: { top: '12px', left: '12px' },
    bottomright: { bottom: '16px', right: '12px' },
    bottomleft: { bottom: '16px', left: '12px' }
  };

  return (
    <div
      style={{
        position: 'absolute',
        ...positionStyles[position],
        zIndex: 1000,
        backgroundColor: 'rgba(255, 255, 255, 0.94)',
        backdropFilter: 'blur(8px)',
        border: '1px solid #cbd5e1',
        borderRadius: '8px',
        padding: '8px 12px',
        boxShadow: '0 4px 14px rgba(0, 30, 60, 0.12)',
        fontSize: '11px',
        display: 'flex',
        flexDirection: 'column',
        gap: '5px',
        pointerEvents: 'auto'
      }}
    >
      <div style={{ fontWeight: 700, color: '#003366', fontSize: '10.5px', textTransform: 'uppercase', letterSpacing: '0.04em', borderBottom: '1px solid #e2e8f0', paddingBottom: '3px', marginBottom: '2px' }}>
        Map Telemetry
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#0284c7', display: 'inline-block', boxShadow: '0 0 4px #0284c7' }} />
        <span style={{ color: '#1e293b', fontWeight: 600 }}>Camera Node</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#138808', display: 'inline-block' }} />
        <span style={{ color: '#475569' }}>Normal Flow</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#f59e0b', display: 'inline-block' }} />
        <span style={{ color: '#475569' }}>Congestion</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#dc2626', display: 'inline-block' }} />
        <span style={{ color: '#475569' }}>Incident</span>
      </div>
    </div>
  );
};

export default MapLegend;
