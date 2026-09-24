import React, { useState } from 'react';

export default function Modals({
  activeModal,
  closeModal,
  onAddNotification
}) {
  const [advisoryCorridor, setAdvisoryCorridor] = useState('All Coimbatore Corridors');
  const [advisoryType, setAdvisoryType] = useState('Traffic Congestion');
  const [advisoryMessage, setAdvisoryMessage] = useState('Heavy congestion observed near Gandhipuram Central. Diversion via Avinashi Road recommended.');
  
  const [overrideCorridor, setOverrideCorridor] = useState('Avinashi Road Express Corridor (CAM_01 to KMCH)');
  const [officerPin, setOfficerPin] = useState('');
  const [overrideError, setOverrideError] = useState('');

  if (!activeModal) return null;

  const handleBroadcast = (e) => {
    e.preventDefault();
    onAddNotification(`Advisory Broadcasted to ${advisoryCorridor}: "${advisoryMessage.slice(0, 45)}..."`);
    closeModal();
  };

  const handleEmergencyOverride = (e) => {
    e.preventDefault();
    if (officerPin.length < 4) {
      setOverrideError('Please enter a valid 4-digit Officer Security PIN.');
      return;
    }
    setOverrideError('');
    onAddNotification(`EMERGENCY GREEN WAVE ACTIVATED: ${overrideCorridor}. Multi-factor audit log #NETRA-OV-${Math.floor(Math.random()*9000+1000)} generated.`);
    closeModal();
    setOfficerPin('');
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      zIndex: 100,
      backgroundColor: 'rgba(7, 28, 54, 0.65)',
      backdropFilter: 'blur(6px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '1rem'
    }}>
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '16px',
        width: '100%',
        maxWidth: '520px',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.25)',
        border: '1px solid rgba(214, 227, 255, 0.8)',
        overflow: 'hidden',
        animation: 'fadeIn 0.2s ease-out'
      }}>
        {/* Top tri-color strip */}
        <div className="tricorn-ribbon" style={{ height: '4px', width: '100%' }} />

        {/* Modal 1: Broadcast Advisory */}
        {activeModal === 'advisory' && (
          <div style={{ padding: '1.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '10px',
                  backgroundColor: '#fff4ea',
                  color: '#ea580c',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>campaign</span>
                </div>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#001e40' }}>Broadcast Traffic Advisory</h3>
                  <p style={{ fontSize: '12px', color: '#64748b' }}>Dispatch urgent alerts to variable messaging signboards (VMS)</p>
                </div>
              </div>
              <button
                onClick={closeModal}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8', padding: '4px' }}
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <form onSubmit={handleBroadcast}>
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                  Target Corridor / Display Network
                </label>
                <select
                  value={advisoryCorridor}
                  onChange={(e) => setAdvisoryCorridor(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.6rem 0.75rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '13px',
                    outline: 'none'
                  }}
                >
                  <option>All Coimbatore Corridors (142 Display Boards)</option>
                  <option>Avinashi Road Express Corridor (CAM_01)</option>
                  <option>Gandhipuram Central Hub (CAM_02)</option>
                  <option>Sathy Road Egress Corridor (CAM_03)</option>
                  <option>Trichy Road Arterial Loop</option>
                </select>
              </div>

              <div style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                  Advisory Severity
                </label>
                <select
                  value={advisoryType}
                  onChange={(e) => setAdvisoryType(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.6rem 0.75rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '13px',
                    outline: 'none'
                  }}
                >
                  <option>Traffic Congestion &amp; Speed Advisory</option>
                  <option>Accident / Lane Clearance Alert</option>
                  <option>Weather / Dense Fog Visibility Warning</option>
                  <option>Emergency Route Diversion</option>
                </select>
              </div>

              <div style={{ marginBottom: '1.25rem' }}>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                  Signboard Message
                </label>
                <textarea
                  rows={3}
                  value={advisoryMessage}
                  onChange={(e) => setAdvisoryMessage(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.6rem 0.75rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '13px',
                    outline: 'none',
                    resize: 'none'
                  }}
                  required
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                <button
                  type="button"
                  onClick={closeModal}
                  style={{
                    padding: '0.5rem 1rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    backgroundColor: '#ffffff',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  style={{
                    padding: '0.5rem 1.25rem',
                    borderRadius: '8px',
                    border: 'none',
                    backgroundColor: '#003366',
                    color: '#ffffff',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>send</span>
                  <span>Dispatch Advisory</span>
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Modal 2: Emergency Override */}
        {activeModal === 'override' && (
          <div style={{ padding: '1.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '10px',
                  backgroundColor: '#fee2e2',
                  color: '#dc2626',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>warning</span>
                </div>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#991b1b' }}>Emergency Signal Override</h3>
                  <p style={{ fontSize: '12px', color: '#64748b' }}>High-priority manual override for emergency response vehicles</p>
                </div>
              </div>
              <button
                onClick={closeModal}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8', padding: '4px' }}
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div style={{
              padding: '0.75rem 1rem',
              backgroundColor: '#fff1f2',
              borderRadius: '8px',
              border: '1px solid #fecdd3',
              fontSize: '12px',
              color: '#9f1239',
              marginBottom: '1rem',
              display: 'flex',
              gap: '8px'
            }}>
              <span className="material-symbols-outlined" style={{ fontSize: '18px', flexShrink: 0 }}>security</span>
              <span>
                WARNING: Signal overrides trigger automatic audit logging to the National Transport Directorate. Unauthorized overrides carry disciplinary penalties under IT Act.
              </span>
            </div>

            <form onSubmit={handleEmergencyOverride}>
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                  Target Green Corridor
                </label>
                <select
                  value={overrideCorridor}
                  onChange={(e) => setOverrideCorridor(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.6rem 0.75rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '13px',
                    outline: 'none'
                  }}
                >
                  <option>Avinashi Road Express Corridor (CAM_01 to KMCH)</option>
                  <option>Gandhipuram Central Arterial (CAM_02 to Medical College)</option>
                  <option>Sathy Road Egress Corridor (CAM_03)</option>
                  <option>Trichy Road Emergency Transit Route</option>
                </select>
              </div>

              <div style={{ marginBottom: '1.25rem' }}>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '4px' }}>
                  Enter Officer Security PIN (4-Digits)
                </label>
                <input
                  type="password"
                  maxLength={6}
                  value={officerPin}
                  onChange={(e) => setOfficerPin(e.target.value)}
                  placeholder="••••"
                  style={{
                    width: '100%',
                    padding: '0.6rem 0.75rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '16px',
                    letterSpacing: '0.25em',
                    outline: 'none'
                  }}
                  required
                />
                {overrideError && (
                  <div style={{ color: '#dc2626', fontSize: '11px', marginTop: '4px' }}>{overrideError}</div>
                )}
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                <button
                  type="button"
                  onClick={closeModal}
                  style={{
                    padding: '0.5rem 1rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    backgroundColor: '#ffffff',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  style={{
                    padding: '0.5rem 1.25rem',
                    borderRadius: '8px',
                    border: 'none',
                    backgroundColor: '#dc2626',
                    color: '#ffffff',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>bolt</span>
                  <span>Authorize Green Wave</span>
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Modal 3: Compliance Report Generation */}
        {activeModal === 'compliance' && (
          <div style={{ padding: '1.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '10px',
                  backgroundColor: '#e0f2fe',
                  color: '#0284c7',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>description</span>
                </div>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#001e40' }}>SIH 2026 Compliance Audit</h3>
                  <p style={{ fontSize: '12px', color: '#64748b' }}>Generate certified government mobility telemetry audit</p>
                </div>
              </div>
              <button
                onClick={closeModal}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8', padding: '4px' }}
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div style={{
              backgroundColor: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '1rem',
              marginBottom: '1.25rem',
              fontSize: '12px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: '#64748b' }}>Audit Reference:</span>
                <span style={{ fontWeight: 600, color: '#003366', fontFamily: 'var(--font-mono)' }}>SIH-2026-AUDIT-992</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: '#64748b' }}>Telemetry Node:</span>
                <span style={{ fontWeight: 600, color: '#003366' }}>Coimbatore Central Cluster #01</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ color: '#64748b' }}>ANPR Precision Score:</span>
                <span style={{ fontWeight: 700, color: '#16a34a' }}>99.2% (ISO/IEC 27001)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#64748b' }}>Digital Signature:</span>
                <span style={{ fontWeight: 600, color: '#0284c7' }}>NIC-SHA256 Certified</span>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button
                type="button"
                onClick={closeModal}
                style={{
                  padding: '0.5rem 1rem',
                  borderRadius: '8px',
                  border: '1px solid #cbd5e1',
                  backgroundColor: '#ffffff',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                Close
              </button>
              <button
                type="button"
                onClick={() => {
                  onAddNotification('SIH 2026 Compliance Audit PDF exported to your local session downloads.');
                  closeModal();
                }}
                style={{
                  padding: '0.5rem 1.25rem',
                  borderRadius: '8px',
                  border: 'none',
                  backgroundColor: '#0284c7',
                  color: '#ffffff',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>download</span>
                <span>Export Official PDF</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
