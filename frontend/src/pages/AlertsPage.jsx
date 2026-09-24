import React, { useState, useEffect } from 'react';
import { getAlerts } from '../services/gisService';

export default function AlertsPage({ onAddNotification }) {
  const [alerts, setAlerts] = useState([
    { id: 'ALT-901', type: 'Overspeeding (>110 km/h)', plate: 'DL 04 C 8891', location: 'Avinashi Road Ingress Node (CAM_01)', severity: 'Critical', time: '14:23 IST', status: 'Pending Review' },
    { id: 'ALT-902', type: 'Wrong Way Arterial Ingress', plate: 'UP 16 F 3301', location: 'Gandhipuram Central Underpass (CAM_02)', severity: 'Critical', time: '14:20 IST', status: 'Pending Review' },
    { id: 'ALT-903', type: 'Stalled Heavy Freight - Lane Block', plate: 'HR 26 DK 5092', location: 'Trichy Road Arterial Junction', severity: 'High', time: '14:18 IST', status: 'Under Investigation' },
    { id: 'ALT-904', type: 'Non-Standard Obscured Number Plate', plate: 'BR 01 AB 1234', location: 'Sathy Road Egress Node (CAM_03)', severity: 'Medium', time: '14:12 IST', status: 'Pending Review' },
    { id: 'ALT-905', type: 'Red Light Jump Violation', plate: 'KA 01 M 9912', location: 'Avinashi Road Express Flyover', severity: 'Medium', time: '14:02 IST', status: 'e-Challan Queued' }
  ]);

  useEffect(() => {
    getAlerts().then((data) => {
      if (data && data.length > 0) {
        setAlerts(data);
      }
    });
  }, []);

  const handleResolve = (id, type) => {
    setAlerts(alerts.filter(a => a.id !== id));
    if (onAddNotification) {
      onAddNotification(`Alert ${id} (${type}) actioned and recorded in shift log.`);
    }
  };

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: '#00677d', letterSpacing: '0.06em' }}>
            Enforcement Dispatch Queue
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: 800, color: '#001e40', margin: 0 }}>
            Active Traffic Alerts &amp; Critical Violations
          </h1>
          <p style={{ fontSize: '12.5px', color: '#64748b', margin: '2px 0 0 0' }}>
            Real-time optical violation detections requiring highway patrol dispatch or e-Challan certification
          </p>
        </div>

        <span style={{
          backgroundColor: '#fef2f2',
          color: '#991b1b',
          border: '1px solid #fecaca',
          padding: '6px 12px',
          borderRadius: '6px',
          fontSize: '12px',
          fontWeight: 700,
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <span style={{ width: '7px', height: '7px', borderRadius: '9999px', backgroundColor: '#dc2626' }} className="animate-pulse-subtle" />
          <span>{alerts.length} Critical Alerts Pending Review</span>
        </span>
      </div>

      {/* Alerts Table */}
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
              <th style={{ padding: '10px 14px' }}>Alert ID</th>
              <th style={{ padding: '10px 12px' }}>Infraction Type</th>
              <th style={{ padding: '10px 12px' }}>Plate Number</th>
              <th style={{ padding: '10px 12px' }}>Corridor Location</th>
              <th style={{ padding: '10px 12px' }}>Severity</th>
              <th style={{ padding: '10px 12px' }}>Time</th>
              <th style={{ padding: '10px 14px', textAlign: 'right' }}>Officer Actions</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((alt) => (
              <tr
                key={alt.id}
                style={{ borderBottom: '1px solid #f1f5f9', transition: 'background-color 0.1s' }}
                onMouseOver={(e) => { e.currentTarget.style.backgroundColor = '#f8fafc'; }}
                onMouseOut={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
              >
                <td style={{ padding: '10px 14px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#003366' }}>{alt.id}</span>
                </td>
                <td style={{ padding: '10px 12px', fontWeight: 700, color: '#0f172a' }}>
                  {alt.type}
                </td>
                <td style={{ padding: '10px 12px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, backgroundColor: '#f1f5f9', padding: '2px 6px', borderRadius: '4px', border: '1px solid #cbd5e1' }}>
                    {alt.plate}
                  </span>
                </td>
                <td style={{ padding: '10px 12px', color: '#475569' }}>
                  {alt.location}
                </td>
                <td style={{ padding: '10px 12px' }}>
                  <span style={{
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: 700,
                    backgroundColor: alt.severity === 'Critical' ? '#fee2e2' : alt.severity === 'High' ? '#ffedd5' : '#fef3c7',
                    color: alt.severity === 'Critical' ? '#991b1b' : alt.severity === 'High' ? '#c2410c' : '#92400e'
                  }}>
                    {alt.severity}
                  </span>
                </td>
                <td style={{ padding: '10px 12px', color: '#64748b' }}>
                  {alt.time}
                </td>
                <td style={{ padding: '10px 14px', textAlign: 'right' }}>
                  <div style={{ display: 'inline-flex', gap: '6px' }}>
                    <button
                      onClick={() => handleResolve(alt.id, alt.type)}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: '#003366',
                        color: '#ffffff',
                        border: 'none',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 600,
                        cursor: 'pointer'
                      }}
                    >
                      Issue e-Challan
                    </button>
                    <button
                      onClick={() => handleResolve(alt.id, alt.type)}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: '#16a34a',
                        color: '#ffffff',
                        border: 'none',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 600,
                        cursor: 'pointer'
                      }}
                    >
                      Clear
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
