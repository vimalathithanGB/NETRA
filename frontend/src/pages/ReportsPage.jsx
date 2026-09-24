import React, { useState } from 'react';
import { complianceReports } from '../data/mockData';

export default function ReportsPage({ onAddNotification }) {
  const [incidents, setIncidents] = useState([
    { id: 'INC-7801', title: 'Port Access Lane 2 Stalled Heavy Freight', corridor: 'Port Access Arterial', severity: 'High', status: 'Pending Review', time: '14:18 IST' },
    { id: 'INC-7802', title: 'High Density Bottleneck at Cyber City Toll Plaza', corridor: 'NH-48 Corridor', severity: 'Medium', status: 'Pending Review', time: '14:05 IST' },
    { id: 'INC-7803', title: 'Two-Wheeler Breakdown on AIIMS Flyover Shoulder', corridor: 'Central Ring Road', severity: 'Low', status: 'Under Review', time: '13:52 IST' },
    { id: 'INC-7804', title: 'Signal Sensor Communication Jitter', corridor: 'DND Flyway Entry', severity: 'Medium', status: 'Pending Review', time: '13:40 IST' }
  ]);

  const handleResolveIncident = (id, title) => {
    setIncidents(incidents.filter(i => i.id !== id));
    onAddNotification(`Incident ${id} (${title}) marked as resolved and logged to shift audit.`);
  };

  return (
    <div className="container-7xl" style={{ paddingTop: '1.75rem', paddingBottom: '3rem' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1.5rem',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div>
          <span style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#00677d' }}>
            Enforcement &amp; Audit Logs
          </span>
          <h1 style={{ fontSize: '24px', fontWeight: 800, color: 'var(--primary-container, #003366)', letterSpacing: '-0.02em' }}>
            Incident Queue &amp; Compliance Reports
          </h1>
          <p style={{ fontSize: '13px', color: '#64748b' }}>
            Officer action queue for field incidents and official SIH 2026 regulatory records
          </p>
        </div>

        <button
          onClick={() => onAddNotification('Downloading master audit zip archive with cryptographic checksums...')}
          style={{
            backgroundColor: '#003366',
            color: '#ffffff',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '8px',
            fontSize: '13px',
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            boxShadow: '0 2px 6px rgba(0, 51, 102, 0.2)'
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>download</span>
          <span>Download Master Archive</span>
        </button>
      </div>

      {/* Incident Review Queue */}
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '16px',
        padding: '1.5rem',
        border: '1px solid #e2e8f0',
        boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)',
        marginBottom: '1.75rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#001e40' }}>Active Incident Review Queue</h3>
            <p style={{ fontSize: '12px', color: '#64748b' }}>Requiring officer acknowledgement and highway patrol dispatch</p>
          </div>
          <span style={{
            padding: '2px 8px',
            backgroundColor: '#fef2f2',
            color: '#991b1b',
            border: '1px solid #fecaca',
            borderRadius: '9999px',
            fontSize: '11px',
            fontWeight: 700
          }}>
            {incidents.length} Action Pending
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {incidents.map((incident) => (
            <div
              key={incident.id}
              style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '10px',
                padding: '1rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '12px'
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '12px', color: '#003366', backgroundColor: '#e0f2fe', padding: '2px 6px', borderRadius: '4px' }}>
                    {incident.id}
                  </span>
                  <span style={{
                    padding: '2px 6px',
                    borderRadius: '4px',
                    fontSize: '10.5px',
                    fontWeight: 700,
                    backgroundColor: incident.severity === 'High' ? '#fee2e2' : incident.severity === 'Medium' ? '#fef3c7' : '#ecfdf5',
                    color: incident.severity === 'High' ? '#991b1b' : incident.severity === 'Medium' ? '#92400e' : '#065f46'
                  }}>
                    {incident.severity} Severity
                  </span>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>{incident.time}</span>
                </div>
                <div style={{ fontSize: '14px', fontWeight: 600, color: '#0f172a' }}>
                  {incident.title}
                </div>
                <div style={{ fontSize: '12px', color: '#475569' }}>
                  Corridor: {incident.corridor}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={() => onAddNotification(`Highway Interceptor Patrol Dispatched to ${incident.corridor}.`)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '6px',
                    backgroundColor: '#ffffff',
                    border: '1px solid #cbd5e1',
                    fontSize: '12px',
                    fontWeight: 600,
                    color: '#334155',
                    cursor: 'pointer'
                  }}
                >
                  Dispatch Patrol
                </button>
                <button
                  onClick={() => handleResolveIncident(incident.id, incident.title)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '6px',
                    backgroundColor: '#16a34a',
                    border: 'none',
                    fontSize: '12px',
                    fontWeight: 700,
                    color: '#ffffff',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>check</span>
                  <span>Clear &amp; Close</span>
                </button>
              </div>
            </div>
          ))}

          {incidents.length === 0 && (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#16a34a', fontWeight: 600, fontSize: '14px' }}>
              All incidents currently resolved. Central Grid operational at zero active warnings.
            </div>
          )}
        </div>
      </div>

      {/* Compliance Reports Archive */}
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '16px',
        padding: '1.5rem',
        border: '1px solid #e2e8f0',
        boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)'
      }}>
        <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#001e40', marginBottom: '4px' }}>
          Certified Regulatory Reports Archive
        </h3>
        <p style={{ fontSize: '12px', color: '#64748b', marginBottom: '1.25rem' }}>
          Digitally signed audit records generated for MoRTH and SIH 2026 Smart Mobility Mission
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {complianceReports.map((report) => (
            <div
              key={report.id}
              style={{
                padding: '12px 14px',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '10px'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span className="material-symbols-outlined" style={{ color: '#0284c7', fontSize: '24px' }}>picture_as_pdf</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '13.5px', color: '#0f172a' }}>{report.title}</div>
                  <div style={{ fontSize: '11.5px', color: '#64748b' }}>
                    Reference: <span style={{ fontFamily: 'var(--font-mono)' }}>{report.id}</span> • Signed by {report.officer} on {report.date}
                  </div>
                </div>
              </div>

              <button
                onClick={() => onAddNotification(`Downloading certified audit ${report.id} (PDF)...`)}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  backgroundColor: '#f1f5f9',
                  border: '1px solid #cbd5e1',
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#003366',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>download</span>
                <span>Download PDF</span>
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
