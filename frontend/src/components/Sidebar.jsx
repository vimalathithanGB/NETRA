import React from 'react';

export default function Sidebar({ currentRoute, setRoute, currentOfficer, onLogout }) {
  const menuItems = [
    { path: '/dashboard', label: 'Dashboard', icon: 'dashboard' },
    { path: '/cameras', label: 'Live Cameras', icon: 'videocam' },
    { path: '/anpr', label: 'ANPR Detection', icon: 'document_scanner' },
    { path: '/tracking', label: 'Vehicle Tracking', icon: 'radar' },
    { path: '/analytics', label: 'Traffic Analytics', icon: 'analytics' },
    { path: '/alerts', label: 'Alerts & Violations', icon: 'warning', badge: '14' },
    { path: '/reports', label: 'Reports', icon: 'assignment' },
    { path: '/cameras-management', label: 'Camera Management', icon: 'settings_suggest' },
    { path: '/settings', label: 'Settings', icon: 'settings' }
  ];

  return (
    <aside style={{
      width: '260px',
      minWidth: '260px',
      backgroundColor: '#0a192f',
      color: '#e2e8f0',
      borderRight: '1px solid #1e293b',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      minHeight: '100vh',
      zIndex: 40,
      flexShrink: 0
    }}>
      {/* Sidebar Header: NETRA Branding */}
      <div>
        <div style={{
          padding: '1.25rem 1.25rem 1rem 1.25rem',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)'
        }}>
          {/* Saffron-White-Green Micro Indicator */}
          <div style={{ display: 'flex', width: '100%', height: '3px', marginBottom: '12px', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ backgroundColor: '#ff9933', width: '33.33%', height: '100%' }} />
            <div style={{ backgroundColor: '#ffffff', width: '33.34%', height: '100%' }} />
            <div style={{ backgroundColor: '#138808', width: '33.33%', height: '100%' }} />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '38px',
              height: '38px',
              borderRadius: '6px',
              backgroundColor: '#003366',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
              boxShadow: '0 2px 6px rgba(0, 0, 0, 0.4)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              position: 'relative',
              flexShrink: 0
            }}>
              <span style={{ fontWeight: 900, fontSize: '16px', letterSpacing: '0.04em' }}>IN</span>
              <div style={{
                position: 'absolute',
                top: '-2px',
                right: '-2px',
                width: '8px',
                height: '8px',
                borderRadius: '9999px',
                backgroundColor: '#ff7a1a',
                border: '1px solid #ffffff'
              }} />
            </div>

            <div>
              <div style={{
                fontSize: '18px',
                fontWeight: 900,
                color: '#ffffff',
                letterSpacing: '0.04em',
                lineHeight: 1.1
              }}>
                NETRA
              </div>
              <div style={{
                fontSize: '9.5px',
                color: '#94a3b8',
                fontWeight: 500,
                lineHeight: 1.2,
                marginTop: '2px'
              }}>
                Networked Engine for Traffic Recognition &amp; Analytics
              </div>
            </div>
          </div>

          <div style={{
            marginTop: '10px',
            fontSize: '9.5px',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            color: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            padding: '3px 8px',
            borderRadius: '4px',
            border: '1px solid rgba(56, 189, 248, 0.2)',
            display: 'inline-block'
          }}>
            MoRTH Command Console
          </div>
        </div>

        {/* Navigation items displayed vertically */}
        <div style={{ padding: '0.75rem 0.6rem' }}>
          <div style={{
            fontSize: '10px',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: '#64748b',
            padding: '0 0.65rem 0.5rem 0.65rem'
          }}>
            Navigation
          </div>

          <nav style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {menuItems.map((item) => {
              const isActive = currentRoute === item.path || (item.path === '/dashboard' && currentRoute === '/');
              return (
                <button
                  key={item.path}
                  onClick={() => setRoute(item.path)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    width: '100%',
                    padding: '0.6rem 0.75rem',
                    borderRadius: '6px',
                    fontSize: '12.5px',
                    fontWeight: isActive ? 700 : 500,
                    color: isActive ? '#ffffff' : '#cbd5e1',
                    backgroundColor: isActive ? '#003366' : 'transparent',
                    border: 'none',
                    borderLeft: isActive ? '3px solid #ff9933' : '3px solid transparent',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    textAlign: 'left'
                  }}
                  onMouseOver={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.06)';
                      e.currentTarget.style.color = '#ffffff';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.color = '#cbd5e1';
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span className="material-symbols-outlined" style={{
                      fontSize: '17px',
                      color: isActive ? '#38bdf8' : '#94a3b8'
                    }}>
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </div>

                  {item.badge && (
                    <span style={{
                      fontSize: '10px',
                      fontWeight: 700,
                      backgroundColor: '#dc2626',
                      color: '#ffffff',
                      padding: '1px 6px',
                      borderRadius: '9999px'
                    }}>
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Sidebar Footer: Officer Profile & Logout */}
      <div style={{
        padding: '1rem',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        backgroundColor: 'rgba(0, 0, 0, 0.15)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '4px',
              backgroundColor: '#003366',
              color: '#ffffff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid rgba(255, 255, 255, 0.2)'
            }}>
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>person</span>
            </div>
            <div>
              <div style={{ fontSize: '11.5px', fontWeight: 700, color: '#ffffff', lineHeight: 1.2 }}>
                {currentOfficer?.name || 'Chief Officer'}
              </div>
              <div style={{ fontSize: '10px', color: '#94a3b8' }}>
                {currentOfficer?.badgeId || 'GOV-IN-7741'}
              </div>
            </div>
          </div>

          <div style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#16a34a' }} className="animate-pulse-subtle" title="Officer Session Active" />
        </div>

        <button
          onClick={onLogout}
          style={{
            width: '100%',
            padding: '0.45rem',
            backgroundColor: 'rgba(220, 38, 38, 0.12)',
            color: '#f87171',
            border: '1px solid rgba(220, 38, 38, 0.3)',
            borderRadius: '5px',
            fontSize: '11.5px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            transition: 'background-color 0.15s'
          }}
          onMouseOver={(e) => { e.currentTarget.style.backgroundColor = 'rgba(220, 38, 38, 0.25)'; }}
          onMouseOut={(e) => { e.currentTarget.style.backgroundColor = 'rgba(220, 38, 38, 0.12)'; }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>logout</span>
          <span>Sign Out of Portal</span>
        </button>
      </div>
    </aside>
  );
}
