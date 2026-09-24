import React, { useState, useEffect } from 'react';

export default function Header({
  language,
  setLanguage,
  fontScale,
  setFontScale,
  highContrast,
  setHighContrast,
  currentOfficer,
  onLogout
}) {
  const [currentTime, setCurrentTime] = useState('');

  // Live IST Clock
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      // Format: DD-MMM-YYYY | HH:mm:ss IST
      const options = {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      };
      const dateStr = now.toLocaleDateString('en-GB', options).replace(',', '');
      setCurrentTime(`${dateStr} IST`);
    };

    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header style={{
      position: 'sticky',
      top: 0,
      left: 0,
      right: 0,
      width: '100%',
      zIndex: 50,
      backgroundColor: '#ffffff',
      borderBottom: '1px solid #d0dbe7',
      boxShadow: '0 1px 4px rgba(0, 30, 60, 0.05)'
    }}>
      {/* 1. Indian Tricolor Thin Line at the very top */}
      <div style={{ display: 'flex', width: '100%', height: '3px' }}>
        <div style={{ backgroundColor: '#ff9933', width: '33.33%', height: '100%' }} />
        <div style={{ backgroundColor: '#ffffff', width: '33.34%', height: '100%' }} />
        <div style={{ backgroundColor: '#138808', width: '33.33%', height: '100%' }} />
      </div>

      {/* 2. Top Utility Bar (Government of India / MoRTH / Time / Accessibility / Language) */}
      <div style={{
        backgroundColor: '#0a192f',
        color: '#e2e8f0',
        fontSize: '11px',
        padding: '3px 0',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <div className="container-gov" style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          {/* Left: Government of India | MoRTH | Official Portal */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 700, letterSpacing: '0.04em', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#ff9933' }}>भारत सरकार</span>
              <span style={{ color: '#94a3b8' }}>|</span>
              <span>Government of India</span>
            </span>
            <span style={{ color: '#475569' }}>•</span>
            <span style={{ color: '#cbd5e1', fontWeight: 500 }}>
              Ministry of Road Transport and Highways (MoRTH)
            </span>
            <span style={{
              backgroundColor: 'rgba(255, 153, 51, 0.2)',
              color: '#ffb366',
              border: '1px solid rgba(255, 153, 51, 0.4)',
              padding: '1px 6px',
              borderRadius: '3px',
              fontSize: '10px',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Official Portal
            </span>
          </div>

          {/* Right: Date/Time | Language | Accessibility | Secure Connection */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            {/* Live Clock */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#94a3b8', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '13px', color: '#38bdf8' }}>schedule</span>
              <span>{currentTime || '19-Sep-2026 | 09:30:00 IST'}</span>
            </div>

            <span style={{ color: '#334155' }}>|</span>

            {/* Language Selector: English / हिंदी */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px' }}>
              <button
                onClick={() => setLanguage && setLanguage('en')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: language === 'hi' ? '#94a3b8' : '#ffffff',
                  fontWeight: language === 'hi' ? 400 : 700,
                  cursor: 'pointer',
                  padding: '1px 3px',
                  textDecoration: language === 'hi' ? 'none' : 'underline'
                }}
              >
                English
              </button>
              <span style={{ color: '#64748b' }}>/</span>
              <button
                onClick={() => setLanguage && setLanguage('hi')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: language === 'hi' ? '#ffffff' : '#94a3b8',
                  fontWeight: language === 'hi' ? 700 : 400,
                  cursor: 'pointer',
                  padding: '1px 3px',
                  textDecoration: language === 'hi' ? 'underline' : 'none'
                }}
              >
                हिंदी
              </button>
            </div>

            <span style={{ color: '#334155' }}>|</span>

            {/* Accessibility: A- A A+ and Contrast */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <button
                onClick={() => setFontScale && setFontScale('sm')}
                title="Decrease Text Size"
                style={{
                  background: fontScale === 'sm' ? '#334155' : 'rgba(255,255,255,0.08)',
                  border: '1px solid #475569',
                  color: '#ffffff',
                  fontSize: '10px',
                  padding: '1px 5px',
                  borderRadius: '3px',
                  cursor: 'pointer'
                }}
              >
                A-
              </button>
              <button
                onClick={() => setFontScale && setFontScale('md')}
                title="Normal Text Size"
                style={{
                  background: fontScale === 'md' ? '#334155' : 'rgba(255,255,255,0.08)',
                  border: '1px solid #475569',
                  color: '#ffffff',
                  fontSize: '10px',
                  padding: '1px 5px',
                  borderRadius: '3px',
                  cursor: 'pointer'
                }}
              >
                A
              </button>
              <button
                onClick={() => setFontScale && setFontScale('lg')}
                title="Increase Text Size"
                style={{
                  background: fontScale === 'lg' ? '#334155' : 'rgba(255,255,255,0.08)',
                  border: '1px solid #475569',
                  color: '#ffffff',
                  fontSize: '10px',
                  padding: '1px 5px',
                  borderRadius: '3px',
                  cursor: 'pointer'
                }}
              >
                A+
              </button>
              <button
                onClick={() => setHighContrast && setHighContrast(!highContrast)}
                title="Toggle High Contrast"
                style={{
                  background: highContrast ? '#ff9933' : 'rgba(255,255,255,0.08)',
                  border: '1px solid #475569',
                  color: highContrast ? '#000000' : '#ffffff',
                  padding: '1px 4px',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>contrast</span>
              </button>
            </div>

            <span style={{ color: '#334155' }}>|</span>

            {/* Secure Connection & Authorized Officer Indicator */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#4ade80' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '13px' }}>lock</span>
                <span style={{ fontSize: '10.5px', fontWeight: 600 }}>TLS 1.3 Secure</span>
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                backgroundColor: 'rgba(56, 189, 248, 0.15)',
                color: '#7dd3fc',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                padding: '1px 6px',
                borderRadius: '3px',
                fontSize: '10.5px',
                fontWeight: 600
              }}>
                <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>verified_user</span>
                <span>Restricted Gov Access</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Second Branding Row (Emblem + NETRA Branding + 24x7 Control Room & SIH 2026) */}
      <div style={{
        padding: '0.6rem 0',
        backgroundColor: '#ffffff'
      }}>
        <div className="container-gov" style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem'
        }}>
          {/* LEFT: Official Department Emblem + NETRA Logo & Text */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* State Emblem of India Lion Capital Motif Icon */}
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '6px',
              backgroundColor: '#f8fafc',
              border: '1px solid #cbd5e1',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
              flexShrink: 0
            }} title="State Emblem of India • सत्यमेव जयते">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L9 7H15L12 2Z" fill="#003366" />
                <path d="M5 8C5 6.5 7 6 7 6L8 9H4C4 9 4.5 8.3 5 8Z" fill="#003366" />
                <path d="M19 8C19 6.5 17 6 17 6L16 9H20C20 9 19.5 8.3 19 8Z" fill="#003366" />
                <rect x="7" y="10" width="10" height="6" rx="1" fill="#003366" />
                <circle cx="12" cy="13" r="2" fill="#ffffff" />
                <rect x="4" y="17" width="16" height="2.5" rx="0.5" fill="#003366" />
                <rect x="6" y="20" width="12" height="1.5" rx="0.5" fill="#ff9933" />
              </svg>
              <span style={{ fontSize: '7px', fontWeight: 800, color: '#003366', marginTop: '1px', letterSpacing: '0.02em' }}>सत्यमेव जयते</span>
            </div>

            {/* NETRA Official Logo Box */}
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '6px',
              backgroundColor: '#003366',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
              boxShadow: '0 2px 6px rgba(0, 51, 102, 0.25)',
              border: '1px solid #002244',
              position: 'relative',
              flexShrink: 0
            }}>
              <span style={{ fontWeight: 900, fontSize: '17px', letterSpacing: '0.04em' }}>IN</span>
              <div style={{
                position: 'absolute',
                top: '-3px',
                right: '-3px',
                width: '10px',
                height: '10px',
                borderRadius: '9999px',
                backgroundColor: '#ff7a1a',
                border: '1.5px solid #ffffff'
              }} />
            </div>

            {/* Text Hierarchy */}
            <div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                <h1 style={{
                  fontSize: '21px',
                  fontWeight: 900,
                  color: '#001e40',
                  letterSpacing: '0.02em',
                  margin: 0,
                  lineHeight: 1.1
                }}>
                  NETRA
                </h1>
                <span style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#334155'
                }}>
                  Networked Engine for Traffic Recognition &amp; Analytics
                </span>
              </div>
              <div style={{
                fontSize: '11px',
                color: '#00677d',
                fontWeight: 600,
                marginTop: '1px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span>Traffic &amp; Transport Intelligence Division</span>
                <span style={{ color: '#cbd5e1' }}>•</span>
                <span style={{ color: '#475569', fontWeight: 500 }}>MoRTH Central Command</span>
              </div>
            </div>
          </div>

          {/* RIGHT: SIH 2026 | Smart Mobility | 24x7 Control Room | Help | Profile */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
            {/* SIH 2026 Badge */}
            <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '2px 8px',
                backgroundColor: '#f0f9ff',
                border: '1px solid #bae6fd',
                color: '#0369a1',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: 700
              }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#0284c7' }} className="animate-pulse-subtle" />
                <span>SIH 2026</span>
              </div>
              <span style={{ fontSize: '10px', color: '#64748b', fontWeight: 600, marginTop: '1px' }}>
                Smart Mobility Mission
              </span>
            </div>

            <div style={{ height: '32px', width: '1px', backgroundColor: '#e2e8f0' }} />

            {/* 24x7 Control Room Status */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', fontSize: '11px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <span style={{
                  width: '7px',
                  height: '7px',
                  borderRadius: '9999px',
                  backgroundColor: '#16a34a',
                  boxShadow: '0 0 6px #16a34a'
                }} className="animate-pulse-subtle" />
                <span style={{ fontWeight: 700, color: '#166534' }}>24x7 Control Room</span>
              </div>
              <span style={{ fontSize: '10.5px', color: '#64748b' }}>
                Toll-Free: <strong>1033</strong> | 1800-NETRA
              </span>
            </div>

            <div style={{ height: '32px', width: '1px', backgroundColor: '#e2e8f0' }} />

            {/* Officer Profile & Sign out */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{
                width: '34px',
                height: '34px',
                borderRadius: '4px',
                backgroundColor: '#003366',
                color: '#ffffff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 1px 3px rgba(0, 30, 60, 0.2)',
                border: '1px solid #002244',
                cursor: 'pointer'
              }} title={currentOfficer ? `${currentOfficer.name} (${currentOfficer.badgeId})` : 'Authorized Officer'}>
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>person</span>
              </div>

              {currentOfficer && (
                <button
                  onClick={onLogout}
                  title="Sign out of portal session"
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#64748b',
                    cursor: 'pointer',
                    padding: '3px',
                    display: 'flex',
                    alignItems: 'center',
                    borderRadius: '4px'
                  }}
                  onMouseOver={(e) => { e.currentTarget.style.color = '#ba1a1a'; }}
                  onMouseOut={(e) => { e.currentTarget.style.color = '#64748b'; }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>logout</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
