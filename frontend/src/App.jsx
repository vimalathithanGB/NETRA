import React, { useState, useEffect } from 'react';
import LoginHeader from './components/LoginHeader';
import AppHeader from './components/AppHeader';
import Sidebar from './components/Sidebar';
import Footer from './components/Footer';
import Modals from './components/Modals';

// Pages
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import CamerasPage from './pages/CamerasPage';
import AnprPage from './pages/AnprPage';
import TrackingPage from './pages/TrackingPage';
import AnalyticsPage from './pages/AnalyticsPage';
import AlertsPage from './pages/AlertsPage';
import ReportsPage from './pages/ReportsPage';
import TelemetryPage from './pages/TelemetryPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  // Determine initial route: Default to /login
  const getInitialRoute = () => {
    const path = window.location.pathname;
    const validRoutes = [
      '/login',
      '/dashboard',
      '/cameras',
      '/anpr',
      '/tracking',
      '/analytics',
      '/alerts',
      '/reports',
      '/cameras-management',
      '/settings'
    ];
    return validRoutes.includes(path) ? path : '/login';
  };

  const [currentRoute, setCurrentRoute] = useState(getInitialRoute());
  const [currentOfficer, setCurrentOfficer] = useState(null);
  const [language, setLanguage] = useState('en'); // 'en' or 'hi'
  const [fontScale, setFontScale] = useState('md'); // 'sm', 'md', 'lg'
  const [highContrast, setHighContrast] = useState(false);
  const [activeModal, setActiveModal] = useState(null); // 'advisory' | 'override' | 'compliance' | null
  const [notification, setNotification] = useState(null);

  // Synchronize browser history and route changes
  const navigate = (newRoute) => {
    setCurrentRoute(newRoute);
    if (window.location.pathname !== newRoute) {
      window.history.pushState({}, '', newRoute);
    }
  };

  // Listen for browser back/forward buttons
  useEffect(() => {
    const handlePopState = () => {
      setCurrentRoute(window.location.pathname || '/login');
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const showNotification = (msg) => {
    setNotification(msg);
    setTimeout(() => {
      setNotification(null);
    }, 4500);
  };

  const handleLoginSuccess = (officer) => {
    setCurrentOfficer(officer);
    navigate('/dashboard');
    showNotification(`Welcome back, ${officer.name}. Session authenticated.`);
  };

  const handleLogout = () => {
    setCurrentOfficer(null);
    navigate('/login');
    showNotification('Officer signed out. Portal locked.');
  };

  // Contrast mode syncing
  useEffect(() => {
    if (highContrast) {
      document.documentElement.setAttribute('data-contrast', 'high');
    } else {
      document.documentElement.removeAttribute('data-contrast');
    }
  }, [highContrast]);

  const isLoginPage = currentRoute === '/login';

  return (
    <div
      className={`font-scale-${fontScale}`}
      style={{
        minHeight: '100vh',
        backgroundColor: 'transparent',
        color: '#071c36',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      {/* Toast Notification Alert Banner */}
      {notification && (
        <div style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          zIndex: 9999,
          backgroundColor: '#001e40',
          color: '#ffffff',
          padding: '10px 16px',
          borderRadius: '6px',
          boxShadow: '0 8px 24px rgba(0, 30, 60, 0.35)',
          border: '1px solid rgba(56, 189, 248, 0.5)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '12.5px',
          fontWeight: 600,
          animation: 'fadeIn 0.2s ease-out',
          maxWidth: '450px'
        }}>
          <span className="material-symbols-outlined" style={{ fontSize: '18px', color: '#38bdf8' }}>info</span>
          <span>{notification}</span>
        </div>
      )}

      {/* ================================================== */}
      {/* STATE 1: CLEAN INDEPENDENT LOGIN PAGE              */}
      {/* ================================================== */}
      {isLoginPage ? (
        <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          {/* Dedicated Login Header (No dashboard navigation, no sidebar) */}
          <LoginHeader
            language={language}
            setLanguage={setLanguage}
            fontScale={fontScale}
            setFontScale={setFontScale}
          />

          {/* Centered Login Card with Abstract Subtle NETRA Background */}
          <main style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
            <LoginPage
              onLoginSuccess={handleLoginSuccess}
            />
          </main>

          {/* Official Government Footer */}
          <Footer />
        </div>
      ) : (
        /* ================================================== */
        /* STATE 2: MAIN NETRA APPLICATION (LEFT SIDEBAR)    */
        /* ================================================== */
        <div style={{ display: 'flex', minHeight: '100vh', width: '100%' }}>
          {/* Left Vertical Sidebar */}
          <Sidebar
            currentRoute={currentRoute}
            setRoute={navigate}
            currentOfficer={currentOfficer}
            onLogout={handleLogout}
          />

          {/* Main Workspace (To the Right of the Sidebar) */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            flexGrow: 1,
            minWidth: 0,
            overflowX: 'hidden'
          }}>
            {/* Top Compact Government Header above main content */}
            <AppHeader
              language={language}
              setLanguage={setLanguage}
              fontScale={fontScale}
              setFontScale={setFontScale}
              highContrast={highContrast}
              setHighContrast={setHighContrast}
              currentOfficer={currentOfficer}
              onLogout={handleLogout}
            />

            {/* Main Content Area rendering active route */}
            <main style={{ flexGrow: 1, backgroundColor: 'transparent' }}>
              {(currentRoute === '/dashboard' || currentRoute === '/') && (
                <DashboardPage
                  onOpenModal={setActiveModal}
                  onAddNotification={showNotification}
                  setRoute={navigate}
                />
              )}

              {currentRoute === '/cameras' && (
                <CamerasPage />
              )}

              {currentRoute === '/anpr' && (
                <AnprPage />
              )}

              {currentRoute === '/tracking' && (
                <TrackingPage />
              )}

              {currentRoute === '/analytics' && (
                <AnalyticsPage />
              )}

              {currentRoute === '/alerts' && (
                <AlertsPage
                  onAddNotification={showNotification}
                />
              )}

              {currentRoute === '/reports' && (
                <ReportsPage
                  onAddNotification={showNotification}
                />
              )}

              {currentRoute === '/cameras-management' && (
                <TelemetryPage />
              )}

              {currentRoute === '/settings' && (
                <SettingsPage
                  onAddNotification={showNotification}
                />
              )}
            </main>

            {/* Government Footer */}
            <Footer />
          </div>
        </div>
      )}

      {/* Interactive Command Toolbar Modals */}
      <Modals
        activeModal={activeModal}
        closeModal={() => setActiveModal(null)}
        onAddNotification={showNotification}
      />
    </div>
  );
}
