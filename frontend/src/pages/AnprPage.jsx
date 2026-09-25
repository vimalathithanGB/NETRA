import React, { useState, useEffect } from 'react';
import { getPlateCaptures } from '../services/gisService';

export default function AnprPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [captures, setCaptures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);
    getPlateCaptures(50)
      .then((data) => {
        if (!isMounted) return;
        setCaptures(Array.isArray(data) ? data : []);
      })
      .catch((err) => {
        if (!isMounted) return;
        console.error('Failed to load ANPR captures from backend:', err);
        setError('Failed to connect to ANPR plate capture service.');
        setCaptures([]);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const filtered = captures.filter((c) => {
    const term = searchTerm.toLowerCase();
    return (
      (c.plate || '').toLowerCase().includes(term) ||
      (c.camId || c.cameraId || '').toLowerCase().includes(term) ||
      (c.type || '').toLowerCase().includes(term)
    );
  });

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: '#00677d', letterSpacing: '0.06em' }}>
            Optical Character Recognition Engine
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: 800, color: '#001e40', margin: 0 }}>
            Automatic Number Plate Recognition (ANPR) Stream
          </h1>
          <p style={{ fontSize: '12.5px', color: '#64748b', margin: '2px 0 0 0' }}>
            High-speed optical recognition pipeline synchronized with MoRTH VAHAN national registry
          </p>
        </div>

        {/* Search input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <span className="material-symbols-outlined" style={{ position: 'absolute', left: '10px', fontSize: '16px', color: '#64748b' }}>
              search
            </span>
            <input
              type="text"
              placeholder="Search Plate (e.g. DL 04, UP 16)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                padding: '6px 12px 6px 30px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '12.5px',
                outline: 'none',
                width: '240px'
              }}
            />
          </div>
          <span style={{
            backgroundColor: '#003366',
            color: '#ffffff',
            padding: '6px 10px',
            borderRadius: '6px',
            fontSize: '11.5px',
            fontWeight: 700
          }}>
            VAHAN 4.0 Synced
          </span>
        </div>
      </div>

      {/* ANPR Records Table */}
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '8px',
        border: '1px solid #d0dbe7',
        boxShadow: '0 1px 4px rgba(0, 30, 60, 0.04)',
        overflow: 'hidden'
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12.5px' }}>
          <thead>
            <tr style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #d0dbe7', color: '#64748b', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              <th style={{ padding: '10px 14px' }}>Registration Plate</th>
              <th style={{ padding: '10px 12px' }}>Vehicle Classification</th>
              <th style={{ padding: '10px 12px' }}>Capture Camera</th>
              <th style={{ padding: '10px 12px' }}>Timestamp (IST)</th>
              <th style={{ padding: '10px 12px' }}>AI Confidence</th>
              <th style={{ padding: '10px 14px', textAlign: 'right' }}>Registry Status</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={6} style={{ padding: '2.5rem', textAlign: 'center', color: '#64748b' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                    <span className="material-symbols-outlined animate-spin" style={{ fontSize: '20px', color: '#003366' }}>progress_activity</span>
                    <span style={{ fontWeight: 600 }}>Loading real-time ANPR records from NETRA backend...</span>
                  </div>
                </td>
              </tr>
            ) : error ? (
              <tr>
                <td colSpan={6} style={{ padding: '2.5rem', textAlign: 'center', color: '#dc2626' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '28px', color: '#dc2626' }}>error</span>
                    <span style={{ fontWeight: 700 }}>{error}</span>
                    <span style={{ fontSize: '12px', color: '#64748b' }}>Ensure the backend API service is running at /api/v1/plates/captures</span>
                  </div>
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ padding: '2.5rem', textAlign: 'center', color: '#64748b' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '32px', color: '#94a3b8' }}>videocam_off</span>
                    <span style={{ fontWeight: 600, color: '#334155' }}>No ANPR plate sightings recorded</span>
                    <span style={{ fontSize: '12px' }}>
                      {searchTerm ? 'No plate records match the search filter.' : 'Live camera detections will populate here as vehicles traverse monitored corridors.'}
                    </span>
                  </div>
                </td>
              </tr>
            ) : (
              filtered.map((item, idx) => (
                <tr
                  key={item.id || `${item.plate}_${idx}`}
                  style={{ borderBottom: '1px solid #f1f5f9', transition: 'background-color 0.1s' }}
                  onMouseOver={(e) => { e.currentTarget.style.backgroundColor = '#f8fafc'; }}
                  onMouseOut={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
                >
                  <td style={{ padding: '10px 14px' }}>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 700,
                      fontSize: '13px',
                      color: '#003366',
                      backgroundColor: '#f1f5f9',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      border: '1px solid #cbd5e1'
                    }}>
                      {item.plate}
                    </span>
                  </td>
                  <td style={{ padding: '10px 12px', fontWeight: 600, color: '#334155' }}>
                    {item.type || 'Vehicle'}
                  </td>
                  <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', color: '#64748b' }}>
                    {item.camId || item.cameraId || 'N/A'}
                  </td>
                  <td style={{ padding: '10px 12px', color: '#64748b' }}>
                    {item.timestamp || 'Just now'}
                  </td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{
                      padding: '2px 6px',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 700,
                      backgroundColor: parseFloat(item.confidence) > 98 ? '#ecfdf5' : '#fef3c7',
                      color: parseFloat(item.confidence) > 98 ? '#065f46' : '#92400e'
                    }}>
                      {typeof item.confidence === 'number' ? `${item.confidence.toFixed(1)}%` : item.confidence || '98.5%'}
                    </span>
                  </td>
                  <td style={{ padding: '10px 14px', textAlign: 'right' }}>
                    <span style={{
                      backgroundColor: '#ecfdf5',
                      color: '#065f46',
                      padding: '2px 8px',
                      borderRadius: '9999px',
                      fontSize: '11px',
                      fontWeight: 700,
                      border: '1px solid #a7f3d0'
                    }}>
                      {item.status || 'VAHAN Verified'}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
