import React from 'react';
import { cameraNodes } from '../data/mockData';

export default function TelemetryPage() {
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
            Infrastructure Diagnostics
          </span>
          <h1 style={{ fontSize: '24px', fontWeight: 800, color: 'var(--primary-container, #003366)', letterSpacing: '-0.02em' }}>
            Edge AI Camera Hardware Telemetry
          </h1>
          <p style={{ fontSize: '13px', color: '#64748b' }}>
            Real-time optical sensor health, inferencing throughput, and hardware telemetry
          </p>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          backgroundColor: '#ecfdf5',
          border: '1px solid #a7f3d0',
          padding: '6px 12px',
          borderRadius: '8px',
          fontSize: '12px',
          fontWeight: 700,
          color: '#065f46'
        }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#059669' }} className="animate-pulse-subtle" />
          <span>Active Nodes: 1,428 / 1,452</span>
        </div>
      </div>

      {/* Nodes Table */}
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '16px',
        padding: '1.5rem',
        border: '1px solid #e2e8f0',
        boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)'
      }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', backgroundColor: '#f8fafc', color: '#64748b', fontSize: '11.5px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                <th style={{ padding: '12px 14px', fontWeight: 700 }}>Node Identifier</th>
                <th style={{ padding: '12px 14px', fontWeight: 700 }}>Corridor Location</th>
                <th style={{ padding: '12px 14px', fontWeight: 700 }}>Throughput (FPS)</th>
                <th style={{ padding: '12px 14px', fontWeight: 700 }}>Inference Latency</th>
                <th style={{ padding: '12px 14px', fontWeight: 700 }}>Edge Neural Model</th>
                <th style={{ padding: '12px 14px', fontWeight: 700, textAlign: 'right' }}>Status</th>
              </tr>
            </thead>
            <tbody style={{ color: '#0f172a' }}>
              {cameraNodes.map((node) => (
                <tr
                  key={node.id}
                  style={{ borderBottom: '1px solid #f1f5f9', transition: 'background-color 0.1s' }}
                  onMouseOver={(e) => { e.currentTarget.style.backgroundColor = '#f0f9ff'; }}
                  onMouseOut={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
                >
                  <td style={{ padding: '12px 14px' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--primary-container, #003366)', backgroundColor: '#f1f5f9', padding: '3px 6px', borderRadius: '4px' }}>
                      {node.id}
                    </span>
                  </td>
                  <td style={{ padding: '12px 14px', fontWeight: 500 }}>
                    {node.location}
                  </td>
                  <td style={{ padding: '12px 14px', fontWeight: 600 }}>
                    <span style={{ color: node.fps >= 50 ? '#16a34a' : '#d97706' }}>
                      {node.fps} FPS
                    </span>
                  </td>
                  <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
                    {node.latency}
                  </td>
                  <td style={{ padding: '12px 14px', fontSize: '12px', color: '#475569' }}>
                    {node.edgeModel}
                  </td>
                  <td style={{ padding: '12px 14px', textAlign: 'right' }}>
                    <span style={{
                      padding: '3px 8px',
                      borderRadius: '9999px',
                      fontSize: '11px',
                      fontWeight: 700,
                      backgroundColor: node.status === 'Active' ? '#ecfdf5' : '#fef3c7',
                      color: node.status === 'Active' ? '#065f46' : '#92400e',
                      border: `1px solid ${node.status === 'Active' ? '#a7f3d0' : '#fde68a'}`
                    }}>
                      {node.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
