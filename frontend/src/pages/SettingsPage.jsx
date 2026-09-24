import React, { useState } from 'react';

export default function SettingsPage({ onAddNotification }) {
  const [minConfidence, setMinConfidence] = useState('95');
  const [encryptionStandard] = useState('256-Bit TLS / AES-GCM (ISO 27001)');
  const [autoDispatch, setAutoDispatch] = useState(true);
  const [retentionDays, setRetentionDays] = useState('90');

  const handleSave = (e) => {
    e.preventDefault();
    if (onAddNotification) {
      onAddNotification('Government command settings updated and digitally signed by Officer.');
    }
  };

  return (
    <div style={{ padding: '1.5rem 2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: '#00677d', letterSpacing: '0.06em' }}>
            System Administration
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: 800, color: '#001e40', margin: 0 }}>
            Command Console &amp; Algorithm Configuration
          </h1>
          <p style={{ fontSize: '12.5px', color: '#64748b', margin: '2px 0 0 0' }}>
            MoRTH compliance standards, edge AI detection parameters, and statutory audit configurations
          </p>
        </div>
      </div>

      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '8px',
        border: '1px solid #d0dbe7',
        padding: '1.5rem',
        maxWidth: '720px',
        boxShadow: '0 1px 4px rgba(0, 30, 60, 0.04)'
      }}>
        <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div>
            <label style={{ display: 'block', fontSize: '12.5px', fontWeight: 700, color: '#001e40', marginBottom: '4px' }}>
              Minimum ANPR Confidence for Automated e-Challan (%)
            </label>
            <input
              type="number"
              min="85"
              max="99"
              value={minConfidence}
              onChange={(e) => setMinConfidence(e.target.value)}
              style={{
                width: '100%',
                maxWidth: '220px',
                padding: '6px 10px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '13px'
              }}
            />
            <p style={{ fontSize: '11px', color: '#64748b', margin: '4px 0 0 0' }}>
              Captures with confidence below this threshold are routed to the manual Officer Review Queue.
            </p>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12.5px', fontWeight: 700, color: '#001e40', marginBottom: '4px' }}>
              CERT-In Cryptographic Protocol Standard
            </label>
            <input
              type="text"
              disabled
              value={encryptionStandard}
              style={{
                width: '100%',
                padding: '6px 10px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                backgroundColor: '#f1f5f9',
                fontSize: '12.5px',
                color: '#475569'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12.5px', fontWeight: 700, color: '#001e40', marginBottom: '4px' }}>
              Statutory Shift Audit Retention Window
            </label>
            <select
              value={retentionDays}
              onChange={(e) => setRetentionDays(e.target.value)}
              style={{
                width: '100%',
                maxWidth: '280px',
                padding: '6px 10px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '12.5px'
              }}
            >
              <option value="90">90 Days (Statutory IT Act Standard)</option>
              <option value="180">180 Days (High Court Judicial Directive)</option>
              <option value="365">1 Year (Long-Term Corridor Study)</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={autoDispatch}
                onChange={(e) => setAutoDispatch(e.target.checked)}
                style={{ accentColor: '#003366', width: '16px', height: '16px' }}
              />
              <span style={{ fontSize: '13px', fontWeight: 600, color: '#1e293b' }}>
                Enable Instant Interceptor Radio Relay for Critical Incidents
              </span>
            </label>
          </div>

          <div style={{ paddingTop: '0.75rem', borderTop: '1px solid #e2e8f0' }}>
            <button
              type="submit"
              style={{
                backgroundColor: '#003366',
                color: '#ffffff',
                border: 'none',
                padding: '8px 18px',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              Save Configuration
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
