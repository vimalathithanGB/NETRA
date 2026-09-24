import React, { useState } from 'react';

export default function LoginPage({ onLoginSuccess }) {
  const [email, setEmail] = useState('officer.netra@morth.gov.in');
  const [password, setPassword] = useState('GovSecure#2026');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberDevice, setRememberDevice] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      onLoginSuccess({
        email,
        name: 'Chief Traffic Officer (Tamil Nadu)',
        badgeId: 'GOV-IN-7741',
        role: 'Authorized Commander'
      });
    }, 450);
  };

  const handleSsoLogin = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      onLoginSuccess({
        email: 'parichay.officer@nic.in',
        name: 'Superintendent R. Sharma',
        badgeId: 'PARICHAY-SSO-9081',
        role: 'National Highway Inspector'
      });
    }, 450);
  };

  return (
    <div style={{
      width: '100%',
      minHeight: 'calc(100vh - 150px)',
      backgroundColor: 'transparent',
      position: 'relative',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2.5rem 1rem'
    }}>

      {/* ================================================== */}
      {/* CENTERED GOVERNMENT LOGIN CARD                     */}
      {/* ================================================== */}
      <div style={{
        position: 'relative',
        zIndex: 10,
        width: '100%',
        maxWidth: '440px',
        margin: '0 auto'
      }}>
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          boxShadow: '0 2px 12px rgba(0, 30, 60, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04)',
          border: '1px solid #d0dbe7',
          overflow: 'hidden'
        }}>
          {/* Card Top Indian Tricolor Strip */}
          <div style={{ display: 'flex', width: '100%', height: '3.5px' }}>
            <div style={{ backgroundColor: '#ff9933', width: '33.33%', height: '100%' }} />
            <div style={{ backgroundColor: '#ffffff', width: '33.34%', height: '100%' }} />
            <div style={{ backgroundColor: '#138808', width: '33.33%', height: '100%' }} />
          </div>

          <div style={{ padding: '1.75rem 1.75rem 1.5rem 1.75rem' }}>
            {/* Card Heading & Subtitle */}
            <div style={{ textAlign: 'center', marginBottom: '1.25rem' }}>
              <div style={{
                fontSize: '10.5px',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
                color: '#00677d',
                marginBottom: '4px'
              }}>
                Government of India • MoRTH
              </div>

              <h2 style={{
                fontSize: '22px',
                fontWeight: 800,
                color: '#001e40',
                letterSpacing: '-0.02em',
                margin: '0 0 4px 0'
              }}>
                Welcome to NETRA
              </h2>

              <p style={{
                fontSize: '12.5px',
                color: '#475569',
                margin: 0
              }}>
                Sign in to access the Traffic Intelligence Portal
              </p>
            </div>

            {/* Restricted Access Badge Above Form */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              backgroundColor: '#f0f4f9',
              padding: '6px 10px',
              borderRadius: '6px',
              marginBottom: '1.25rem',
              fontSize: '11.5px',
              fontWeight: 700,
              color: '#003366',
              border: '1px solid #d0dbe7'
            }}>
              <span className="material-symbols-outlined" style={{ fontSize: '15px', color: '#0057b8' }}>verified_user</span>
              <span>Restricted Access • Authorized Personnel Only</span>
            </div>

            {/* Login Form */}
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {/* Field 1: Official Email ID / Officer ID */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#1e293b',
                  marginBottom: '4px'
                }}>
                  Official Email ID / Officer ID
                </label>
                <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                  <span className="material-symbols-outlined" style={{
                    position: 'absolute',
                    left: '10px',
                    color: '#64748b',
                    fontSize: '16px'
                  }}>
                    badge
                  </span>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    style={{
                      width: '100%',
                      padding: '0.65rem 0.85rem 0.65rem 2.25rem',
                      backgroundColor: '#ffffff',
                      color: '#071c36',
                      borderRadius: '6px',
                      border: '1px solid #cbd5e1',
                      fontSize: '13px',
                      outline: 'none',
                      transition: 'border-color 0.15s'
                    }}
                    onFocus={(e) => { e.currentTarget.style.borderColor = '#003366'; }}
                    onBlur={(e) => { e.currentTarget.style.borderColor = '#cbd5e1'; }}
                    placeholder="officer.id@gov.in"
                  />
                </div>
              </div>

              {/* Field 2: Secure Password / Passkey */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#1e293b',
                  marginBottom: '4px'
                }}>
                  Secure Password / Passkey
                </label>
                <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                  <span className="material-symbols-outlined" style={{
                    position: 'absolute',
                    left: '10px',
                    color: '#64748b',
                    fontSize: '16px'
                  }}>
                    lock
                  </span>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    style={{
                      width: '100%',
                      padding: '0.65rem 2.25rem 0.65rem 2.25rem',
                      backgroundColor: '#ffffff',
                      color: '#071c36',
                      borderRadius: '6px',
                      border: '1px solid #cbd5e1',
                      fontSize: '13px',
                      outline: 'none'
                    }}
                    onFocus={(e) => { e.currentTarget.style.borderColor = '#003366'; }}
                    onBlur={(e) => { e.currentTarget.style.borderColor = '#cbd5e1'; }}
                    placeholder="••••••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    style={{
                      position: 'absolute',
                      right: '10px',
                      background: 'none',
                      border: 'none',
                      color: '#64748b',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      padding: 0
                    }}
                    title={showPassword ? 'Hide password' : 'Show password'}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                      {showPassword ? 'visibility_off' : 'visibility'}
                    </span>
                  </button>
                </div>
              </div>

              {/* Actions: Remember device & Forgot Password? */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: '12px',
                marginTop: '1px'
              }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', color: '#475569' }}>
                  <input
                    type="checkbox"
                    checked={rememberDevice}
                    onChange={(e) => setRememberDevice(e.target.checked)}
                    style={{ accentColor: '#003366', width: '14px', height: '14px', cursor: 'pointer' }}
                  />
                  <span>Remember device</span>
                </label>
                <a
                  href="#forgot"
                  onClick={(e) => {
                    e.preventDefault();
                    alert('Password Reset Notice:\nPlease contact your Zonal NIC Control Desk (Ext: 1033) or email support-netra@gov.in with your Officer ID.');
                  }}
                  style={{ color: '#0057b8', textDecoration: 'none', fontWeight: 600 }}
                >
                  Forgot Password?
                </a>
              </div>

              {/* Primary Button: Sign In to Portal */}
              <button
                type="submit"
                disabled={isLoading}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  backgroundColor: '#003366',
                  color: '#ffffff',
                  fontSize: '13.5px',
                  fontWeight: 700,
                  borderRadius: '6px',
                  border: 'none',
                  boxShadow: '0 2px 6px rgba(0, 51, 102, 0.2)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  marginTop: '0.35rem',
                  transition: 'background-color 0.15s ease'
                }}
                onMouseOver={(e) => { e.currentTarget.style.backgroundColor = '#002244'; }}
                onMouseOut={(e) => { e.currentTarget.style.backgroundColor = '#003366'; }}
              >
                <span>{isLoading ? 'Verifying Credentials...' : 'Sign In to Portal'}</span>
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>arrow_forward</span>
              </button>
            </form>

            {/* Divider: OR */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              margin: '1rem 0',
              opacity: 0.8
            }}>
              <div style={{ flexGrow: 1, height: '1px', backgroundColor: '#e2e8f0' }} />
              <span style={{
                margin: '0 10px',
                fontSize: '10.5px',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: '#94a3b8'
              }}>
                OR
              </span>
              <div style={{ flexGrow: 1, height: '1px', backgroundColor: '#e2e8f0' }} />
            </div>

            {/* Institutional Google Account Button */}
            <button
              type="button"
              onClick={handleSsoLogin}
              style={{
                width: '100%',
                padding: '0.65rem',
                backgroundColor: '#ffffff',
                color: '#0f172a',
                fontSize: '12.5px',
                fontWeight: 600,
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.03)',
                transition: 'background-color 0.15s'
              }}
              onMouseOver={(e) => { e.currentTarget.style.backgroundColor = '#f8fafc'; }}
              onMouseOut={(e) => { e.currentTarget.style.backgroundColor = '#ffffff'; }}
            >
              <svg style={{ width: '16px', height: '16px' }} viewBox="0 0 24 24">
                <path d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.66-5.17 3.66-9.17z" fill="#4285F4" />
                <path d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.13 0-5.78-2.11-6.73-4.96H1.19v3.15C3.18 21.35 7.22 24 12 24z" fill="#34A853" />
                <path d="M5.27 14.24c-.25-.72-.38-1.49-.38-2.24s.13-1.52.38-2.24V6.6H1.19C.43 8.13 0 9.87 0 11.7s.43 3.57 1.19 5.1l4.08-2.56z" fill="#FBBC05" />
                <path d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.22 0 3.18 2.65 1.19 6.6l4.08 3.15c.95-2.85 3.6-4.96 6.73-4.96z" fill="#EA4335" />
              </svg>
              <span>Institutional Google Account</span>
            </button>

            {/* ================================================== */}
            {/* 5. GOVERNMENT SECURITY INFORMATION SECTION          */}
            {/* ================================================== */}
            <div style={{
              marginTop: '1.25rem',
              padding: '0.85rem',
              backgroundColor: '#f8fafc',
              borderRadius: '6px',
              border: '1px solid #e2e8f0',
              fontSize: '11px',
              color: '#475569'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontWeight: 700,
                color: '#003366',
                marginBottom: '6px',
                fontSize: '11px',
                textTransform: 'uppercase',
                letterSpacing: '0.04em'
              }}>
                <span className="material-symbols-outlined" style={{ fontSize: '14px', color: '#0057b8' }}>security</span>
                <span>Authorized Personnel Only</span>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '4px 8px',
                marginBottom: '8px',
                fontSize: '10.5px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#166534', fontWeight: 600 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>lock</span>
                  <span>256-bit TLS Encrypted</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#0369a1', fontWeight: 600 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>verified</span>
                  <span>NIC CERT-In Compliant</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#475569', fontWeight: 600, gridColumn: 'span 2' }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>history_edu</span>
                  <span>All activity is logged and cryptographically audited</span>
                </div>
              </div>

              <div style={{
                borderTop: '1px solid #e2e8f0',
                paddingTop: '6px',
                fontSize: '10px',
                color: '#64748b',
                lineHeight: 1.4
              }}>
                <strong>Statutory Notice:</strong> Unauthorized access or tampering is strictly prohibited and punishable under Sections 43 &amp; 66 of the Information Technology Act, 2000.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
