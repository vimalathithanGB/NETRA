import React from 'react';

export default function Navigation({ currentRoute, setRoute }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
    { id: 'analytics', label: 'Analytics', icon: 'analytics' },
    { id: 'tracking', label: 'Vehicle Tracking', icon: 'radar' },
    { id: 'anpr', label: 'ANPR Detection', icon: 'document_scanner' },
    { id: 'traffic-intel', label: 'Traffic Intelligence', icon: 'traffic' },
    { id: 'reports', label: 'Reports', icon: 'assignment' },
    { id: 'camera-mgmt', label: 'Camera Management', icon: 'videocam' },
    { id: 'login', label: 'Auth Portal', icon: 'lock' }
  ];

  return (
    <nav style={{
      backgroundColor: '#ffffff',
      borderBottom: '1px solid #d0dbe7',
      boxShadow: '0 1px 2px rgba(0, 0, 0, 0.03)',
      position: 'relative',
      zIndex: 40
    }}>
      <div className="container-gov" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        overflowX: 'auto',
        whiteSpace: 'nowrap',
        scrollbarWidth: 'none'
      }}>
        {/* Horizontal Navigation List */}
        <div style={{ display: 'flex', alignItems: 'center' }}>
          {navItems.map((item) => {
            const isActive = currentRoute === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setRoute(item.id)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '0.65rem 0.95rem',
                  fontSize: '12.5px',
                  fontWeight: isActive ? 700 : 600,
                  color: isActive ? '#ffffff' : '#001e40',
                  backgroundColor: isActive ? '#003366' : 'transparent',
                  border: 'none',
                  borderBottom: isActive ? '3px solid #ff9933' : '3px solid transparent',
                  cursor: 'pointer',
                  transition: 'background-color 0.15s, color 0.15s',
                  position: 'relative'
                }}
                onMouseOver={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.backgroundColor = '#f1f5f9';
                    e.currentTarget.style.color = '#003366';
                  }
                }}
                onMouseOut={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = '#001e40';
                  }
                }}
              >
                <span className="material-symbols-outlined" style={{
                  fontSize: '15px',
                  color: isActive ? '#ffffff' : '#00677d'
                }}>
                  {item.icon}
                </span>
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>

        {/* Right side live status pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '11px',
          color: '#475569',
          paddingLeft: '1rem',
          flexShrink: 0
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{
              width: '6px',
              height: '6px',
              borderRadius: '9999px',
              backgroundColor: '#16a34a'
            }} className="animate-pulse-subtle" />
            <span style={{ fontWeight: 600, color: '#166534' }}>Grid Node Online</span>
          </div>
          <span style={{ color: '#cbd5e1' }}>|</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: '#003366', fontWeight: 600 }}>
            MoRTH-VNET 2.6
          </span>
        </div>
      </div>
    </nav>
  );
}
