import React, { useState, useEffect } from 'react';
import { getCameras } from '../services/gisService';

export default function CamerasPage() {
  const [selectedCorridor, setSelectedCorridor] = useState('All Corridors');
  const [camList, setCamList] = useState([]);
  const [activeCam, setActiveCam] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch verified camera records from PostgreSQL via FastAPI backend
  useEffect(() => {
    getCameras()
      .then((data) => {
        if (data && data.length > 0) {
          const mapped = data.map((c) => ({
            id: c.id,
            name: c.name || `Camera ${c.id}`,
            location: c.locationName || c.location_name || c.name || `Node ${c.id}`,
            roadName: c.roadName || c.road_name || 'Corridor Link',
            fps: c.fps || 60,
            status: c.status === 'online' ? 'Active' : c.status === 'warning' ? 'Degraded' : 'Offline',
            latency: '< 10ms',
            edgeModel: 'NETRA-YOLO-v11',
            speedLimitKmh: c.speedLimitKmh || c.speed_limit_kmh || 40,
            latitude: c.latitude,
            longitude: c.longitude,
            vehicleCount: c.vehicleCount ?? 0,
            lastDetection: c.lastDetection || 'Active'
          }));
          setCamList(mapped);
          setActiveCam(mapped[0]);
        }
        setIsLoading(false);
      })
      .catch((err) => {
        console.warn('Cameras load exception:', err);
        setIsLoading(false);
      });
  }, []);

  // Compute dynamic corridor filters from real camera records
  const uniqueCorridors = [
    'All Corridors',
    ...Array.from(new Set(camList.map((c) => c.roadName).filter(Boolean)))
  ];

  const filteredCameras = selectedCorridor === 'All Corridors'
    ? camList
    : camList.filter((c) => c.roadName === selectedCorridor);

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      {/* Title Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: '#00677d', letterSpacing: '0.06em' }}>
            Optical Surveillance Network
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: 800, color: '#001e40', margin: 0 }}>
            Live Camera Feeds &amp; Node Matrix
          </h1>
          <p style={{ fontSize: '12.5px', color: '#64748b', margin: '2px 0 0 0' }}>
            Multi-corridor real-time video surveillance feeds with automated vehicle detection overlays
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <select
            value={selectedCorridor}
            onChange={(e) => setSelectedCorridor(e.target.value)}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: '1px solid #cbd5e1',
              fontSize: '12px',
              backgroundColor: '#ffffff',
              color: '#1e293b',
              fontWeight: 600,
              outline: 'none'
            }}
          >
            {uniqueCorridors.map((corridor) => (
              <option key={corridor} value={corridor}>
                {corridor}
              </option>
            ))}
          </select>

          <span style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            backgroundColor: '#ecfdf5',
            color: '#065f46',
            border: '1px solid #a7f3d0',
            padding: '5px 10px',
            borderRadius: '6px',
            fontSize: '11.5px',
            fontWeight: 700
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#059669' }} className="animate-pulse-subtle" />
            <span>{camList.length} Active Feeds</span>
          </span>
        </div>
      </div>

      {isLoading ? (
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          border: '1px solid #d0dbe7',
          padding: '3rem',
          textAlign: 'center',
          color: '#64748b'
        }}>
          Connecting to NETRA Video Matrix &amp; Camera Records...
        </div>
      ) : activeCam ? (
        /* Grid: Main Selected Video Stream + Camera List */
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '1.25rem' }}>
          {/* Left 8 Cols: Main Active Feed Preview */}
          <div style={{
            gridColumn: 'span 8',
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #d0dbe7',
            overflow: 'hidden',
            boxShadow: '0 2px 6px rgba(0, 30, 60, 0.04)'
          }}>
            {/* Feed Header */}
            <div style={{
              padding: '10px 16px',
              backgroundColor: '#0a192f',
              color: '#ffffff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ width: '7px', height: '7px', borderRadius: '9999px', backgroundColor: '#ef4444' }} className="animate-pulse-subtle" />
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{activeCam.id}</strong>
                <span style={{ color: '#94a3b8' }}>•</span>
                <span>{activeCam.location}</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px', color: '#94a3b8' }}>
                <span>FPS: <strong style={{ color: '#4ade80' }}>{activeCam.fps}</strong></span>
                <span>Limit: <strong style={{ color: '#38bdf8' }}>{activeCam.speedLimitKmh} km/h</strong></span>
                <span style={{ backgroundColor: 'rgba(255, 255, 255, 0.1)', padding: '2px 6px', borderRadius: '4px', color: '#ffffff' }}>
                  {activeCam.edgeModel}
                </span>
              </div>
            </div>

            {/* Live Camera Canvas with HUD Overlays */}
            <div style={{
              position: 'relative',
              height: '420px',
              backgroundColor: '#0f172a',
              backgroundImage: `url('https://lh3.googleusercontent.com/aida-public/AB6AXuDocl6ay--j8pqhOmgC9HCjTlFmbxdsck5CVgGzxg7wplMwgxTZTRHrMjvXgL30117PMsVWFTADJrFtAwELPQB8rlYvc6rN3MybElQcwXdXQ6duazVfOzZ2ClifjYbMrh7N1EV5BoOSMzwTzUSog2sVwrsw6K-5oaWPFqBa2pjIpcj6CNzp1wZNmf7WuGo3SgiZBdkiAM1FbyMO2EGpB8toMnaf7_H6YRYV2OQPB9tFdh961ZmwEbeoww')`,
              backgroundSize: 'cover',
              backgroundPosition: 'center'
            }}>
              {/* Dark optical vignette */}
              <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0, 20, 40, 0.35)' }} />

              {/* Top HUD Overlay */}
              <div style={{
                position: 'absolute',
                top: '12px',
                left: '12px',
                backgroundColor: 'rgba(10, 25, 47, 0.85)',
                backdropFilter: 'blur(6px)',
                padding: '6px 12px',
                borderRadius: '4px',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                color: '#ffffff',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)'
              }}>
                <div>REC ● LIVE OPTICAL STREAM</div>
                <div style={{ color: '#38bdf8', fontSize: '10px' }}>
                  NODE: {activeCam.id} | CORRIDOR: {activeCam.roadName}
                </div>
              </div>

              {/* Bottom Real-time Stream Status — Prepared for dynamic AI bounding boxes */}
              <div style={{
                position: 'absolute',
                bottom: '12px',
                left: '12px',
                right: '12px',
                backgroundColor: 'rgba(10, 25, 47, 0.85)',
                backdropFilter: 'blur(6px)',
                padding: '8px 12px',
                borderRadius: '4px',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                color: '#e2e8f0',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '8px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#22c55e' }} className="animate-pulse-subtle" />
                  <span>STREAM READY FOR NODE {activeCam.id} ({activeCam.roadName})</span>
                </div>
                <div style={{ color: '#94a3b8', fontSize: '10.5px' }}>
                  COORDS: {activeCam.latitude?.toFixed(4)}°N, {activeCam.longitude?.toFixed(4)}°E
                </div>
              </div>
            </div>

            {/* Camera Metadata Telemetry Strip */}
            <div style={{
              padding: '12px 16px',
              backgroundColor: '#f8fafc',
              borderTop: '1px solid #e2e8f0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#475569',
              flexWrap: 'wrap',
              gap: '8px'
            }}>
              <div>
                <strong>Road Corridor:</strong> {activeCam.roadName}
              </div>
              <div>
                <strong>Speed Limit:</strong> {activeCam.speedLimitKmh} km/h
              </div>
              <div>
                <strong>Optical Status:</strong> <span style={{ color: '#16a34a', fontWeight: 700 }}>{activeCam.status}</span>
              </div>
            </div>
          </div>

          {/* Right 4 Cols: Real Backend Camera Matrix List */}
          <div style={{
            gridColumn: 'span 4',
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #d0dbe7',
            padding: '1rem',
            boxShadow: '0 2px 6px rgba(0, 30, 60, 0.04)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: '#001e40' }}>
                Select Active Camera Node
              </div>
              <span style={{ fontSize: '11px', color: '#64748b' }}>
                {filteredCameras.length} Nodes
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '460px', overflowY: 'auto' }}>
              {filteredCameras.map((node) => {
                const isSelected = activeCam && activeCam.id === node.id;
                return (
                  <div
                    key={node.id}
                    onClick={() => setActiveCam(node)}
                    style={{
                      padding: '8px 10px',
                      borderRadius: '6px',
                      border: `1px solid ${isSelected ? '#003366' : '#e2e8f0'}`,
                      backgroundColor: isSelected ? '#f0f4f9' : '#ffffff',
                      cursor: 'pointer',
                      transition: 'all 0.15s'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '11.5px', color: '#003366' }}>
                        {node.id}
                      </span>
                      <span style={{
                        fontSize: '10px',
                        fontWeight: 700,
                        color: node.status === 'Active' ? '#166534' : '#92400e',
                        backgroundColor: node.status === 'Active' ? '#dcfce7' : '#fef3c7',
                        padding: '1px 5px',
                        borderRadius: '3px'
                      }}>
                        {node.status}
                      </span>
                    </div>
                    <div style={{ fontSize: '11px', color: '#475569', marginTop: '2px' }}>
                      {node.location}
                    </div>
                    <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '1px' }}>
                      Corridor: {node.roadName}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          border: '1px solid #d0dbe7',
          padding: '3rem',
          textAlign: 'center',
          color: '#64748b'
        }}>
          No active camera nodes found in grid database.
        </div>
      )}
    </div>
  );
}
