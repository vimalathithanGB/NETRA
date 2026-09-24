import React, { useState, useEffect } from 'react';

export default function AppHeader({
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

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
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
      backgroundColor: '#ffffff',
      borderBottom: '1px solid #d0dbe7',
      boxShadow: '0 1px 3px rgba(0, 30, 60, 0.05)',
      position: 'sticky',
      top: 0,
      zIndex: 30,
      width: '100%'
    }}>
      {/* Top Tricolor Accent Stripe */}
      <div style={{ display: 'flex', width: '100%', height: '3px' }}>
        <div style={{ backgroundColor: '#ff9933', width: '33.33%', height: '100%' }} />
        <div style={{ backgroundColor: '#ffffff', width: '33.34%', height: '100%' }} />
        <div style={{ backgroundColor: '#138808', width: '33.33%', height: '100%' }} />
      </div>

      {/* Main Top Header Content */}
      <div style={{
        padding: '0.55rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '0.75rem'
      }}>
        {/* Left: Ministry & Portal Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Emblem Icon */}
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '4px',
            backgroundColor: '#f8fafc',
            border: '1px solid #cbd5e1',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }} title="Government of India">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L9 7H15L12 2Z" fill="#003366" />
              <rect x="7" y="9" width="10" height="6" rx="1" fill="#003366" />
              <circle cx="12" cy="12" r="1.5" fill="#ffffff" />
              <rect x="4" y="16" width="16" height="2" fill="#003366" />
              <rect x="6" y="19" width="12" height="1.5" fill="#ff9933" />
            </svg>
          </div>

          <div>
            <div style={{
              fontSize: '11px',
              fontWeight: 700,
              color: '#00677d',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <span>भारत सरकार</span>
              <span style={{ color: '#cbd5e1' }}>|</span>
              <span>Government of India</span>
              <span style={{ color: '#cbd5e1' }}>•</span>
              <span style={{ color: '#475569', fontWeight: 500 }}>Ministry of Road Transport and Highways (MoRTH)</span>
            </div>
            <div style={{ fontSize: '13px', fontWeight: 800, color: '#001e40' }}>
              National Traffic Command Center Console
            </div>
          </div>
        </div>

        {/* Right: Telemetry Time | 24x7 Control Room | Help | SIH 2026 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap', fontSize: '11px' }}>
          {/* Live Clock */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#475569', fontFamily: 'var(--font-mono)' }}>
            <span className="material-symbols-outlined" style={{ fontSize: '14px', color: '#0284c7' }}>schedule</span>
            <span>{currentTime || '19-Sep-2026 | 09:30:00 IST'}</span>
          </div>

          <span style={{ color: '#cbd5e1' }}>|</span>

          {/* 24x7 Control Room */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{
              width: '7px',
              height: '7px',
              borderRadius: '9999px',
              backgroundColor: '#16a34a',
              boxShadow: '0 0 6px #16a34a'
            }} className="animate-pulse-subtle" />
            <span style={{ fontWeight: 700, color: '#166534' }}>24x7 Control Room Active</span>
          </div>

          <span style={{ color: '#cbd5e1' }}>|</span>

          {/* SIH 2026 Pill */}
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            padding: '2px 8px',
            backgroundColor: '#f0f9ff',
            border: '1px solid #bae6fd',
            color: '#0369a1',
            borderRadius: '4px',
            fontSize: '10.5px',
            fontWeight: 700
          }}>
            <span style={{ width: '5px', height: '5px', borderRadius: '9999px', backgroundColor: '#0284c7' }} className="animate-pulse-subtle" />
            <span>SIH 2026</span>
          </div>

          <span style={{ color: '#cbd5e1' }}>|</span>

          {/* Language & Accessibility Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <button
              onClick={() => setLanguage && setLanguage(language === 'en' ? 'hi' : 'en')}
              style={{
                background: '#f1f5f9',
                border: '1px solid #cbd5e1',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '10.5px',
                fontWeight: 600,
                color: '#334155',
                cursor: 'pointer'
              }}
            >
              {language === 'en' ? 'हिंदी' : 'English'}
            </button>

            <button
              onClick={() => setFontScale && setFontScale(fontScale === 'md' ? 'lg' : 'md')}
              title="Toggle Font Size"
              style={{
                background: '#f1f5f9',
                border: '1px solid #cbd5e1',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '10.5px',
                fontWeight: 700,
                color: '#334155',
                cursor: 'pointer'
              }}
            >
              A{fontScale === 'lg' ? '-' : '+'}
            </button>
          </div>

          {/* Sign out small action */}
          <button
            onClick={onLogout}
            title="Lock & Exit to Login Portal"
            style={{
              background: 'none',
              border: 'none',
              color: '#64748b',
              cursor: 'pointer',
              padding: '2px 4px',
              display: 'flex',
              alignItems: 'center',
              gap: '3px',
              fontSize: '11px',
              fontWeight: 600
            }}
            onMouseOver={(e) => { e.currentTarget.style.color = '#dc2626'; }}
            onMouseOut={(e) => { e.currentTarget.style.color = '#64748b'; }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>lock</span>
            <span>Lock</span>
          </button>
        </div>
      </div>
    </header>
  );
}
