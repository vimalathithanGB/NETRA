import React from 'react';

export default function Footer() {
  const footerLinks = [
    { label: 'Privacy Policy', href: '#privacy' },
    { label: 'Terms of Use', href: '#terms' },
    { label: 'Security Audit', href: '#security' },
    { label: 'Accessibility', href: '#accessibility' },
    { label: 'NIC Guidelines', href: '#guidelines' },
    { label: 'Contact Us', href: '#contact' }
  ];

  return (
    <footer style={{
      width: '100%',
      backgroundColor: '#0a192f',
      color: '#cbd5e1',
      borderTop: '3px solid #003366',
      padding: '1.25rem 0',
      marginTop: 'auto',
      fontSize: '11.5px'
    }}>
      <div className="container-gov" style={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        {/* Left: Department & Title */}
        <div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', marginBottom: '2px' }}>
            <strong style={{ color: '#ffffff', fontSize: '13px', letterSpacing: '0.04em' }}>NETRA</strong>
            <span style={{ color: '#94a3b8' }}>-</span>
            <span style={{ color: '#e2e8f0', fontWeight: 500 }}>
              Networked Engine for Traffic Recognition &amp; Analytics
            </span>
          </div>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>
            Government of India • Ministry of Road Transport and Highways (MoRTH)
          </div>
        </div>

        {/* Right: Official Policy & Compliance Links */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          flexWrap: 'wrap',
          fontWeight: 500
        }}>
          {footerLinks.map((link, idx) => (
            <React.Fragment key={link.label}>
              <a
                href={link.href}
                onClick={(e) => e.preventDefault()}
                style={{
                  color: '#94a3b8',
                  textDecoration: 'none',
                  transition: 'color 0.15s'
                }}
                onMouseOver={(e) => { e.currentTarget.style.color = '#38bdf8'; }}
                onMouseOut={(e) => { e.currentTarget.style.color = '#94a3b8'; }}
              >
                {link.label}
              </a>
              {idx < footerLinks.length - 1 && (
                <span style={{ color: '#334155' }}>|</span>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </footer>
  );
}
