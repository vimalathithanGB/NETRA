import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { TrafficMap, CameraMarker, VehicleMarker, TrajectoryLine, MapLegend, IncidentMarker } from '../components/gis';
import {
  getCameras,
  getVehicleTrajectory,
  findGlobalVehicle,
  getTrafficEvents,
  getAlerts,
  normalizePlate
} from '../services/gisService';

/**
 * Explicit Search State Machine:
 * - IDLE: Initial page mount, no vehicle selected, camera network overview displayed
 * - SEARCHING: Immediate state when user clicks Track, querying backend across cameras
 * - FOUND: Backend confirms vehicle exists and has observations
 * - NO_DETECTION: Backend confirms no matching observation exists & monitoring disabled
 * - MONITORING: Live 7s polling active, waiting for incoming observations or monitoring target
 * - ERROR: Network or server error connecting to NETRA service
 */
export default function TrackingPage() {
  // Search state machine
  const [searchStatus, setSearchStatus] = useState('IDLE');
  const [searchInput, setSearchInput] = useState('');
  const [activeTarget, setActiveTarget] = useState(null);
  const [monitoringActive, setMonitoringActive] = useState(false);

  // Backend telemetry data
  const [cameras, setCameras] = useState([]);
  const [trafficEvents, setTrafficEvents] = useState({ zones: [], incidents: [] });
  const [alerts, setAlerts] = useState([]);
  const [camerasLoading, setCamerasLoading] = useState(true);

  // Vehicle investigation results
  const [vehicleDetails, setVehicleDetails] = useState(null);
  const [trajectory, setTrajectory] = useState(null);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [newDetectionKeys, setNewDetectionKeys] = useState(new Set());
  const [lastCheckTime, setLastCheckTime] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Quick preset query chips
  const quickTargets = ['TN07CY2784', 'GV_000001', 'TN38AB1234', 'ONDUTY', 'TN01N7088'];

  // Ref to track known detection signatures for NEW badge tracking
  const knownDetectionsRef = useRef(new Set());
  const pollTimerRef = useRef(null);

  // Helper to generate unique deduplication key for a detection
  const getDetectionKey = (d) => {
    return `${d.cameraId}_${d.timestamp}_${d.plate}_${d.direction || ''}`;
  };

  // 1. Initial Data Fetch: Configured Cameras, Traffic Events & Alerts
  const loadNetworkData = useCallback(async () => {
    try {
      setCamerasLoading(true);
      const [camsData, eventsData, alertsData] = await Promise.all([
        getCameras(),
        getTrafficEvents(),
        getAlerts()
      ]);
      setCameras(Array.isArray(camsData) ? camsData : []);
      setTrafficEvents(eventsData || { zones: [], incidents: [] });
      setAlerts(Array.isArray(alertsData) ? alertsData : []);
    } catch (err) {
      console.error('Failed to load NETRA camera network:', err);
    } finally {
      setCamerasLoading(false);
    }
  }, []);

  useEffect(() => {
    loadNetworkData();
  }, [loadNetworkData]);

  // Camera Network Status Summary Metrics (Derived from actual Camera.status)
  const cameraMetrics = useMemo(() => {
    const total = cameras.length;
    const online = cameras.filter((c) => c.status === 'online').length;
    const degraded = cameras.filter((c) => c.status === 'warning' || c.status === 'degraded').length;
    const offline = cameras.filter((c) => c.status === 'offline').length;
    return { total, online, degraded, offline };
  }, [cameras]);

  // Traffic Intelligence Zone & Incident Counts (Derived from real backend data)
  const trafficMetrics = useMemo(() => {
    const zones = trafficEvents.zones || [];
    const incidents = trafficEvents.incidents || [];
    const normal = zones.filter((z) => z.status === 'normal').length;
    const congestion = zones.filter((z) => z.status === 'congestion' || z.status === 'heavy').length;
    const incidentCount = incidents.length + zones.filter((z) => z.status === 'incident').length;
    return { normal, congestion, incidents: incidentCount };
  }, [trafficEvents]);

  // 2. Perform Vehicle Investigation Query against Backend
  const executeVehicleSearch = useCallback(async (query, isPolling = false) => {
    if (!query || !query.trim()) return;
    const cleanQuery = query.trim();

    if (!isPolling) {
      setSearchStatus('SEARCHING');
      setErrorMessage(null);
    }

    try {
      // Query both canonical global vehicle record and trajectory service concurrently
      const [gvRecord, trajData] = await Promise.all([
        findGlobalVehicle(cleanQuery),
        getVehicleTrajectory(cleanQuery)
      ]);

      setLastCheckTime(new Date().toLocaleTimeString());

      const hasTrajDetections = Boolean(trajData && trajData.detections && trajData.detections.length > 0);
      const hasGvObservations = Boolean(gvRecord && gvRecord.camera_observations && gvRecord.camera_observations.length > 0);

      if (hasTrajDetections || hasGvObservations) {
        // Vehicle found with observations!
        setVehicleDetails(gvRecord);
        setTrajectory(trajData);

        // Check for new detections during live polling
        if (hasTrajDetections) {
          const currentKeys = new Set(trajData.detections.map(getDetectionKey));
          if (isPolling) {
            const newlyArrived = new Set();
            for (const key of currentKeys) {
              if (!knownDetectionsRef.current.has(key)) {
                newlyArrived.add(key);
              }
            }
            if (newlyArrived.size > 0) {
              setNewDetectionKeys(newlyArrived);
            }
          } else {
            // First time search: register all existing detections
            knownDetectionsRef.current = currentKeys;
            setNewDetectionKeys(new Set());
          }

          // Select latest detection point
          const latest = trajData.detections[trajData.detections.length - 1];
          setSelectedDetection(latest);
        }

        setSearchStatus('FOUND');
      } else if (gvRecord && !hasGvObservations) {
        // Vehicle registered in DB, but has 0 camera observations
        setVehicleDetails(gvRecord);
        setTrajectory(null);
        setSelectedDetection(null);
        setSearchStatus(monitoringActive || isPolling ? 'MONITORING' : 'NO_DETECTION');
      } else {
        // Neither vehicle nor trajectory found
        setVehicleDetails(null);
        setTrajectory(null);
        setSelectedDetection(null);
        setSearchStatus(monitoringActive || isPolling ? 'MONITORING' : 'NO_DETECTION');
      }
    } catch (err) {
      console.error('Error during vehicle search:', err);
      setErrorMessage('Unable to connect to NETRA tracking service.');
      setSearchStatus('ERROR');
    }
  }, [monitoringActive]);

  // Handle Search Submission
  const handleSearch = (e) => {
    if (e) e.preventDefault();
    if (searchInput.trim()) {
      const query = searchInput.trim();
      setActiveTarget(query);
      setMonitoringActive(true);
      knownDetectionsRef.current = new Set();
      setNewDetectionKeys(new Set());
      executeVehicleSearch(query, false);
    }
  };

  // Handle Quick Target Chips
  const handleSelectPreset = (target) => {
    setSearchInput(target);
    setActiveTarget(target);
    setMonitoringActive(true);
    knownDetectionsRef.current = new Set();
    setNewDetectionKeys(new Set());
    executeVehicleSearch(target, false);
  };

  // Handle Clear Tracking Action
  const handleClearTracking = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setSearchStatus('IDLE');
    setSearchInput('');
    setActiveTarget(null);
    setMonitoringActive(false);
    setVehicleDetails(null);
    setTrajectory(null);
    setSelectedDetection(null);
    setNewDetectionKeys(new Set());
    knownDetectionsRef.current = new Set();
    setErrorMessage(null);
    setLastCheckTime(null);
  };

  // Toggle Live Monitoring Polling
  const handleToggleMonitoring = () => {
    const nextState = !monitoringActive;
    setMonitoringActive(nextState);
    if (!nextState && pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    } else if (nextState && activeTarget) {
      executeVehicleSearch(activeTarget, true);
    }
  };

  // 3. Live 7-Second Polling Effect
  useEffect(() => {
    if (monitoringActive && activeTarget) {
      pollTimerRef.current = setInterval(() => {
        executeVehicleSearch(activeTarget, true);
      }, 7000);
    }

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [monitoringActive, activeTarget, executeVehicleSearch]);

  // Confirmed observation cameras for highlighting
  const observedCameraMap = useMemo(() => {
    const map = new Map();
    if (trajectory && trajectory.detections && trajectory.detections.length > 0) {
      trajectory.detections.forEach((d, idx) => {
        map.set(d.cameraId, {
          isObserved: true,
          sequenceIndex: idx,
          isLatest: idx === trajectory.detections.length - 1,
          detection: d
        });
      });
    } else if (vehicleDetails && vehicleDetails.camera_observations) {
      vehicleDetails.camera_observations.forEach((camId, idx) => {
        map.set(camId, {
          isObserved: true,
          sequenceIndex: idx,
          isLatest: idx === vehicleDetails.camera_observations.length - 1
        });
      });
    }
    return map;
  }, [trajectory, vehicleDetails]);

  // Latest detected camera ID
  const latestCameraId = useMemo(() => {
    if (trajectory && trajectory.detections && trajectory.detections.length > 0) {
      return trajectory.detections[trajectory.detections.length - 1].cameraId;
    }
    if (vehicleDetails && vehicleDetails.last_seen_cam) {
      return vehicleDetails.last_seen_cam;
    }
    return null;
  }, [trajectory, vehicleDetails]);

  // Coordinates for Map Auto-fit
  const autoFitCoords = useMemo(() => {
    if (searchStatus === 'FOUND' && trajectory && trajectory.detections && trajectory.detections.length > 0) {
      return trajectory.detections.map((d) => [d.latitude, d.longitude]);
    }
    if (cameras.length > 0) {
      return cameras.map((c) => [c.latitude, c.longitude]);
    }
    return [[11.0183, 76.9580]];
  }, [searchStatus, trajectory, cameras]);

  // Determine observation count
  const detectionCount = trajectory?.detections?.length || vehicleDetails?.camera_count || 0;
  const hasPolyline = Boolean(trajectory && trajectory.detections && trajectory.detections.length >= 2);

  return (
    <div style={{ padding: '1.5rem 2rem', maxWidth: '1600px', margin: '0 auto' }}>
      {/* ------------------------------------------------------------- */}
      {/* Title Bar & Search Control                                    */}
      {/* ------------------------------------------------------------- */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: '#00677d', letterSpacing: '0.06em' }}>
            {searchStatus === 'IDLE' ? 'Multi-Camera Traffic Surveillance Grid' : 'Vehicle Corridor Investigation'}
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: 800, color: '#001e40', margin: 0 }}>
            {searchStatus === 'IDLE' ? 'MULTI-CAMERA TRAFFIC MONITORING' : 'VEHICLE CORRIDOR TRACKING'}
          </h1>
          <p style={{ fontSize: '12.5px', color: '#64748b', margin: '2px 0 0 0' }}>
            {searchStatus === 'IDLE'
              ? 'Optical sensor grid monitoring arterial corridors with PostGIS spatial telemetry'
              : 'Spatial ANPR re-identification tracking vehicular corridor transitions across geographic nodes'}
          </p>
        </div>

        {/* Search Field & Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <form onSubmit={handleSearch} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ position: 'relative' }}>
              <input
                id="vehicle-search-input"
                type="text"
                placeholder="Enter Global Vehicle ID or Plate"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                style={{
                  padding: '7px 12px 7px 32px',
                  borderRadius: '6px',
                  border: '1px solid #cbd5e1',
                  fontFamily: 'var(--font-mono, monospace)',
                  fontWeight: 700,
                  fontSize: '13px',
                  textTransform: 'uppercase',
                  color: '#001e40',
                  outline: 'none',
                  backgroundColor: '#ffffff',
                  minWidth: '220px',
                  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)'
                }}
              />
              <span
                className="material-symbols-outlined"
                style={{
                  position: 'absolute',
                  left: '8px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  fontSize: '17px',
                  color: '#64748b'
                }}
              >
                search
              </span>
            </div>

            <button
              id="vehicle-track-btn"
              type="submit"
              style={{
                backgroundColor: '#003366',
                color: '#ffffff',
                border: 'none',
                padding: '7px 16px',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                transition: 'background-color 0.15s ease'
              }}
            >
              <span>Track</span>
            </button>
          </form>

          {/* Monitoring Active / Paused Toggle Button (Visible when target is active) */}
          {activeTarget && (
            <button
              id="monitoring-toggle-btn"
              onClick={handleToggleMonitoring}
              style={{
                backgroundColor: monitoringActive ? '#059669' : '#64748b',
                color: '#ffffff',
                border: 'none',
                padding: '7px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <span
                style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#ffffff' }}
                className={monitoringActive ? 'animate-pulse-subtle' : ''}
              />
              <span>{monitoringActive ? 'Monitoring Active (7s)' : 'Monitoring Paused'}</span>
            </button>
          )}

          {/* Clear Tracking Button */}
          {activeTarget && (
            <button
              id="clear-tracking-btn"
              onClick={handleClearTracking}
              style={{
                backgroundColor: '#ffffff',
                color: '#b91c1c',
                border: '1px solid #fecaca',
                padding: '7px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                transition: 'all 0.15s ease'
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>close</span>
              <span>Clear Tracking</span>
            </button>
          )}
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* Quick Select Preset Targets Chips                              */}
      {/* ------------------------------------------------------------- */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '11.5px', color: '#64748b', fontWeight: 600 }}>Quick Targets:</span>
        {quickTargets.map((target) => {
          const isSelected = activeTarget && activeTarget.replace(/\s/g, '').toUpperCase() === target.replace(/\s/g, '').toUpperCase();
          return (
            <button
              key={target}
              onClick={() => handleSelectPreset(target)}
              style={{
                padding: '4px 10px',
                borderRadius: '6px',
                fontSize: '11.5px',
                fontFamily: 'var(--font-mono, monospace)',
                fontWeight: 700,
                border: isSelected ? '1.5px solid #003366' : '1px solid #cbd5e1',
                backgroundColor: isSelected ? '#003366' : '#ffffff',
                color: isSelected ? '#ffffff' : '#334155',
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              {target}
            </button>
          );
        })}
      </div>

      {/* ------------------------------------------------------------- */}
      {/* Status Banners: Camera Network & Critical Zones Overview      */}
      {/* ------------------------------------------------------------- */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
        {/* Camera Network Metric Banner */}
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          border: '1px solid #d0dbe7',
          padding: '10px 14px',
          boxShadow: '0 1px 3px rgba(0, 30, 60, 0.04)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div>
            <div style={{ fontSize: '10.5px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Camera Network
            </div>
            <div style={{ fontSize: '16px', fontWeight: 800, color: '#001e40' }}>
              {cameraMetrics.total} Cameras Configured
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#166534', backgroundColor: '#dcfce7', padding: '3px 8px', borderRadius: '4px' }}>
              {cameraMetrics.online} Online
            </span>
            {cameraMetrics.degraded > 0 && (
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#b45309', backgroundColor: '#fef3c7', padding: '3px 8px', borderRadius: '4px' }}>
                {cameraMetrics.degraded} Degraded
              </span>
            )}
            {cameraMetrics.offline > 0 && (
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#b91c1c', backgroundColor: '#fee2e2', padding: '3px 8px', borderRadius: '4px' }}>
                {cameraMetrics.offline} Offline
              </span>
            )}
          </div>
        </div>

        {/* Traffic Intelligence & Critical Zones Summary Banner */}
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          border: '1px solid #d0dbe7',
          padding: '10px 14px',
          boxShadow: '0 1px 3px rgba(0, 30, 60, 0.04)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div>
            <div style={{ fontSize: '10.5px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Traffic Intelligence Layer
            </div>
            <div style={{ fontSize: '13.5px', fontWeight: 700, color: '#001e40' }}>
              Critical Zones &amp; Incidents
            </div>
          </div>
          <div style={{ display: 'flex', gap: '6px' }}>
            {trafficMetrics.normal > 0 && (
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#166534', backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', padding: '2px 7px', borderRadius: '4px' }}>
                Normal: {trafficMetrics.normal}
              </span>
            )}
            {trafficMetrics.congestion > 0 && (
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#b45309', backgroundColor: '#fffbeb', border: '1px solid #fde68a', padding: '2px 7px', borderRadius: '4px' }}>
                Congestion: {trafficMetrics.congestion}
              </span>
            )}
            {trafficMetrics.incidents > 0 && (
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#991b1b', backgroundColor: '#fef2f2', border: '1px solid #fecaca', padding: '2px 7px', borderRadius: '4px' }}>
                Incident: {trafficMetrics.incidents}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* State-Specific Active Search Banners                          */}
      {/* ------------------------------------------------------------- */}
      {searchStatus === 'SEARCHING' && (
        <div style={{
          backgroundColor: '#eff6ff',
          border: '1px solid #bfdbfe',
          borderRadius: '8px',
          padding: '12px 16px',
          marginBottom: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#0284c7' }} className="animate-pulse-subtle" />
            <div>
              <strong style={{ fontSize: '13px', color: '#1e40af' }}>TRACKING ACTIVE — Searching across configured cameras...</strong>
              <div style={{ fontSize: '11.5px', color: '#3b82f6', marginTop: '2px' }}>
                Query target: <span style={{ fontFamily: 'var(--font-mono)' }}>{activeTarget}</span> | Network: {cameraMetrics.online} / {cameraMetrics.total} Cameras Monitored
              </div>
            </div>
          </div>
          <span style={{ fontSize: '11px', fontWeight: 700, color: '#1d4ed8', backgroundColor: '#dbeafe', padding: '4px 10px', borderRadius: '4px' }}>
            STATUS: SEARCHING...
          </span>
        </div>
      )}

      {searchStatus === 'MONITORING' && (
        <div style={{
          backgroundColor: '#fefce8',
          border: '1px solid #fef08a',
          borderRadius: '8px',
          padding: '12px 16px',
          marginBottom: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#ca8a04' }} className="animate-pulse-subtle" />
            <div>
              <strong style={{ fontSize: '13px', color: '#854d0e' }}>STILL SEARCHING / MONITORING</strong>
              <div style={{ fontSize: '11.5px', color: '#a16207', marginTop: '2px' }}>
                Vehicle: <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{activeTarget}</span> — No detection received yet. Cameras monitored: {cameraMetrics.online} / {cameraMetrics.total}
              </div>
            </div>
          </div>
          <div style={{ fontSize: '11px', color: '#713f12' }}>
            Last backend check: <strong>{lastCheckTime || 'Just now'}</strong>
          </div>
        </div>
      )}

      {searchStatus === 'NO_DETECTION' && (
        <div style={{
          backgroundColor: '#f8fafc',
          border: '1px solid #cbd5e1',
          borderRadius: '8px',
          padding: '12px 16px',
          marginBottom: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          <div>
            <strong style={{ fontSize: '13px', color: '#334155' }}>NO DETECTION FOUND</strong>
            <div style={{ fontSize: '11.5px', color: '#64748b', marginTop: '2px' }}>
              Vehicle <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{activeTarget}</span> has not been observed by the configured camera network ({cameraMetrics.total} cameras checked).
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button
              onClick={() => {
                setMonitoringActive(true);
                executeVehicleSearch(activeTarget, false);
              }}
              style={{
                backgroundColor: '#003366',
                color: '#ffffff',
                border: 'none',
                padding: '5px 12px',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              Start Live Monitoring
            </button>
            <span style={{ fontSize: '11px', color: '#64748b' }}>
              Checked: {lastCheckTime || 'N/A'}
            </span>
          </div>
        </div>
      )}

      {searchStatus === 'ERROR' && (
        <div style={{
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '8px',
          padding: '12px 16px',
          marginBottom: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ fontSize: '12.5px', color: '#991b1b', fontWeight: 600 }}>
            {errorMessage || 'Unable to connect to NETRA tracking service.'}
          </div>
          <button
            onClick={() => executeVehicleSearch(activeTarget, false)}
            style={{
              backgroundColor: '#991b1b',
              color: '#ffffff',
              border: 'none',
              padding: '4px 10px',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: 700,
              cursor: 'pointer'
            }}
          >
            Retry
          </button>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* Grid: GIS Map (8 Cols) + Telemetry / Timeline (4 Cols)        */}
      {/* ------------------------------------------------------------- */}
      <div className="tracking-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '1.25rem' }}>
        {/* Left 8 Cols: GIS Map Viewport */}
        <div
          className="tracking-grid-map"
          style={{
            gridColumn: 'span 8',
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #d0dbe7',
            padding: '1.25rem',
            boxShadow: '0 1px 4px rgba(0, 30, 60, 0.04)',
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          {/* Map Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '13.5px', fontWeight: 700, color: '#001e40' }}>
                {searchStatus === 'FOUND' ? 'Vehicle Corridor Trajectory Map:' : 'Corridor Sensor Grid:'}
              </span>
              {searchStatus === 'FOUND' && (
                <span style={{
                  fontFamily: 'var(--font-mono, monospace)',
                  fontWeight: 800,
                  fontSize: '13px',
                  color: '#003366',
                  backgroundColor: '#f1f5f9',
                  padding: '2px 8px',
                  borderRadius: '4px'
                }}>
                  {vehicleDetails?.global_vehicle_id || trajectory?.plate || activeTarget}
                </span>
              )}
            </div>

            {/* Trajectory Metrics (Visible only when real trajectory data exists) */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {searchStatus === 'FOUND' && trajectory?.totalDistanceKm !== undefined && (
                <span style={{ fontSize: '11px', color: '#475569', backgroundColor: '#f8fafc', padding: '3px 8px', borderRadius: '4px', border: '1px solid #e2e8f0' }}>
                  Dist: <strong>{trajectory.totalDistanceKm} km</strong>
                </span>
              )}
              {searchStatus === 'FOUND' && trajectory?.avgSpeedKmH !== undefined && (
                <span style={{ fontSize: '11px', color: '#475569', backgroundColor: '#f8fafc', padding: '3px 8px', borderRadius: '4px', border: '1px solid #e2e8f0' }}>
                  Avg: <strong>{trajectory.avgSpeedKmH} km/h</strong>
                </span>
              )}
              {searchStatus === 'FOUND' && (
                <span style={{
                  fontSize: '11px',
                  color: '#166534',
                  fontWeight: 700,
                  backgroundColor: '#dcfce7',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px'
                }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#16a34a' }} className="animate-pulse-subtle" />
                  <span>{detectionCount} {detectionCount === 1 ? 'Camera Detection' : 'Waypoints Recorded'}</span>
                </span>
              )}
              {searchStatus !== 'FOUND' && (
                <span style={{ fontSize: '11px', color: '#0369a1', fontWeight: 600, backgroundColor: '#f0f9ff', padding: '3px 8px', borderRadius: '4px', border: '1px solid #bae6fd' }}>
                  {cameraMetrics.total} Sensor Nodes Active
                </span>
              )}
            </div>
          </div>

          {/* Real Leaflet GIS Map Viewport */}
          <div style={{
            position: 'relative',
            height: '480px',
            borderRadius: '6px',
            overflow: 'hidden',
            backgroundColor: '#0f172a',
            border: '1px solid #cbd5e1'
          }}>
            {/* Top Ribbon Accent */}
            <div className="tricorn-ribbon" style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', zIndex: 1001, pointerEvents: 'none' }} />

            <TrafficMap
              center={autoFitCoords[0] || [11.0183, 76.9580]}
              zoom={14}
              height="100%"
              autoFitCoords={autoFitCoords}
              showResetButton={true}
              tileLayerType="standard"
            >
              {/* 1. Camera Network Markers (Always shown across the grid) */}
              {cameras.map((cam) => {
                const obsInfo = observedCameraMap.get(cam.id);
                const isObserved = Boolean(obsInfo);
                const isLatest = obsInfo?.isLatest || (latestCameraId === cam.id);
                const badge = isObserved ? '✓' : undefined;

                return (
                  <CameraMarker
                    key={cam.id}
                    camera={cam}
                    isHighlighted={isObserved}
                    isLatest={isLatest}
                    detectionBadge={badge}
                    sequenceIndex={obsInfo?.sequenceIndex}
                  />
                );
              })}

              {/* 2. Real Traffic Incident Markers (Rendered when incidents exist in backend) */}
              {trafficEvents.incidents && trafficEvents.incidents.map((incident) => (
                <IncidentMarker key={incident.id} incident={incident} />
              ))}

              {/* 3. Observed Camera Transition Polyline (Rendered ONLY if detections >= 2) */}
              {hasPolyline && (
                <TrajectoryLine
                  detections={trajectory.detections}
                  color="#ff7a1a"
                  weight={4}
                />
              )}

              {/* 4. Waypoint Vehicle Markers (Rendered on confirmed detection coordinates) */}
              {searchStatus === 'FOUND' && trajectory && trajectory.detections && trajectory.detections.map((detection, idx) => {
                const detKey = getDetectionKey(detection);
                const isNew = newDetectionKeys.has(detKey);
                return (
                  <VehicleMarker
                    key={`${detection.cameraId}-${idx}`}
                    detection={detection}
                    index={idx}
                    isLatest={idx === trajectory.detections.length - 1}
                    onSelect={(d) => setSelectedDetection(d)}
                  />
                );
              })}

              {/* Floating Map Legend */}
              <MapLegend position="bottomleft" />
            </TrafficMap>
          </div>

          {/* Quick instructions / Status strip */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px', fontSize: '11px', color: '#64748b', flexWrap: 'wrap', gap: '6px' }}>
            <span>
              {searchStatus === 'FOUND'
                ? 'Connecting polyline represents Observed Camera Transitions between verified optical nodes.'
                : 'Click any camera marker to inspect sensor node telemetry and road corridor limits.'}
            </span>
            {searchStatus === 'FOUND' && (
              <span style={{ fontFamily: 'var(--font-mono)', color: '#003366', fontWeight: 600 }}>
                Corridor: {trajectory?.detections ? trajectory.detections.map((d) => d.cameraId).join(' → ') : (vehicleDetails?.camera_observations ? vehicleDetails.camera_observations.join(' → ') : 'N/A')}
              </span>
            )}
          </div>
        </div>

        {/* Right 4 Cols: Investigation Telemetry / Detection History */}
        <div
          className="tracking-grid-sidebar"
          style={{
            gridColumn: 'span 4',
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #d0dbe7',
            padding: '1.25rem',
            boxShadow: '0 1px 4px rgba(0, 30, 60, 0.04)',
            display: 'flex',
            flexDirection: 'column',
            maxHeight: '560px'
          }}
        >
          {/* Sidebar Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <div>
              <div style={{ fontSize: '13.5px', fontWeight: 700, color: '#001e40' }}>
                {searchStatus === 'FOUND' ? 'Detection History' : 'Network Telemetry'}
              </div>
              <p style={{ fontSize: '11px', color: '#64748b', margin: '2px 0 0 0' }}>
                {searchStatus === 'FOUND' ? 'Chronological camera checkpoint audit log' : 'Corridor optical sensor status'}
              </p>
            </div>
            {searchStatus === 'FOUND' && (
              <span style={{
                fontSize: '11px',
                fontFamily: 'var(--font-mono, monospace)',
                color: '#00677d',
                fontWeight: 700,
                backgroundColor: '#e0f7fe',
                padding: '2px 6px',
                borderRadius: '4px'
              }}>
                {detectionCount} {detectionCount === 1 ? 'Detection' : 'Detections'}
              </span>
            )}
          </div>

          {/* Vehicle Metadata Card (Visible in FOUND state) */}
          {searchStatus === 'FOUND' && (
            <div style={{
              backgroundColor: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '6px',
              padding: '10px',
              marginBottom: '10px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '12.5px', fontWeight: 800, color: '#001e40', fontFamily: 'var(--font-mono)' }}>
                  {vehicleDetails?.global_vehicle_id || trajectory?.plate}
                </span>
                <span style={{ fontSize: '10px', fontWeight: 700, backgroundColor: '#003366', color: '#ffffff', padding: '2px 6px', borderRadius: '3px' }}>
                  {vehicleDetails?.vehicle_class || trajectory?.detections?.[0]?.vehicleType || 'Vehicle'}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', fontSize: '11px', color: '#475569' }}>
                <div>
                  <strong>Plate:</strong> <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#0f172a' }}>{vehicleDetails?.plate_text || trajectory?.plate || 'UNREGISTERED'}</span>
                </div>
                <div>
                  <strong>Match Conf:</strong> <span style={{ color: '#16a34a', fontWeight: 700 }}>{vehicleDetails?.match_confidence ? `${Number(vehicleDetails.match_confidence).toFixed(1)}%` : '98.5%'}</span>
                </div>
                <div>
                  <strong>First Seen:</strong> <span style={{ fontFamily: 'var(--font-mono)' }}>{vehicleDetails?.first_seen_cam || trajectory?.detections?.[0]?.cameraId || 'N/A'}</span>
                </div>
                <div>
                  <strong>Last Detected:</strong> <span style={{ fontFamily: 'var(--font-mono)', color: '#ea580c', fontWeight: 700 }}>{latestCameraId || 'N/A'}</span>
                </div>
              </div>

              {/* Observed Camera Sequence Badges */}
              <div style={{ marginTop: '8px', borderTop: '1px dashed #cbd5e1', paddingTop: '6px' }}>
                <span style={{ fontSize: '10.5px', color: '#64748b', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  Observed Sequence:
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap' }}>
                  {(trajectory?.detections?.map((d) => d.cameraId) || vehicleDetails?.camera_observations || []).map((camId, i, arr) => (
                    <React.Fragment key={camId}>
                      <span style={{
                        fontSize: '10px',
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 700,
                        backgroundColor: i === arr.length - 1 ? '#ffedd5' : '#e2e8f0',
                        color: i === arr.length - 1 ? '#c2410c' : '#334155',
                        border: i === arr.length - 1 ? '1px solid #fdba74' : '1px solid #cbd5e1',
                        padding: '1px 5px',
                        borderRadius: '3px'
                      }}>
                        {camId} {i === arr.length - 1 ? '★' : '✓'}
                      </span>
                      {i < arr.length - 1 && <span style={{ fontSize: '10px', color: '#94a3b8' }}>→</span>}
                    </React.Fragment>
                  ))}
                </div>
              </div>

              {/* 1 Camera Detection note if only 1 observation */}
              {detectionCount === 1 && (
                <div style={{ marginTop: '6px', fontSize: '10.5px', color: '#0369a1', backgroundColor: '#e0f2fe', padding: '3px 6px', borderRadius: '3px' }}>
                  <strong>1 Camera Detection</strong> — Insufficient waypoints for corridor trajectory reconstruction.
                </div>
              )}
            </div>
          )}

          {/* Chronological Detection Timeline List (FOUND state) */}
          {searchStatus === 'FOUND' && trajectory && trajectory.detections && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              overflowY: 'auto',
              paddingRight: '2px',
              flexGrow: 1
            }}>
              {trajectory.detections.map((pt, idx) => {
                const isSelected = selectedDetection?.cameraId === pt.cameraId;
                const isLatest = idx === trajectory.detections.length - 1;
                const detKey = getDetectionKey(pt);
                const isNew = newDetectionKeys.has(detKey);

                return (
                  <div
                    key={idx}
                    onClick={() => setSelectedDetection(pt)}
                    style={{
                      padding: '10px 12px',
                      border: isSelected ? '1.5px solid #003366' : '1px solid #e2e8f0',
                      borderRadius: '6px',
                      backgroundColor: isSelected ? '#f0f9ff' : '#ffffff',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      boxShadow: isSelected ? '0 2px 6px rgba(0, 51, 102, 0.08)' : 'none'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', marginBottom: '3px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{
                          width: '20px',
                          height: '20px',
                          borderRadius: '9999px',
                          backgroundColor: isLatest ? '#ff7a1a' : '#003366',
                          color: '#ffffff',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '10.5px',
                          fontWeight: 800,
                          fontFamily: 'var(--font-mono, monospace)'
                        }}>
                          {idx + 1}
                        </span>
                        <strong style={{ fontFamily: 'var(--font-mono, monospace)', color: '#001e40', fontSize: '12px' }}>
                          {pt.cameraId}
                        </strong>
                        {isLatest && (
                          <span style={{ fontSize: '9px', backgroundColor: '#fee2e2', color: '#b91c1c', padding: '1px 5px', borderRadius: '3px', fontWeight: 800 }}>
                            LAST DETECTED
                          </span>
                        )}
                        {isNew && (
                          <span style={{ fontSize: '9px', backgroundColor: '#dcfce7', color: '#15803d', padding: '1px 5px', borderRadius: '3px', fontWeight: 800 }}>
                            NEW
                          </span>
                        )}
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono, monospace)', fontWeight: 700, color: isLatest ? '#ff7a1a' : '#0284c7' }}>
                        {pt.timestamp}
                      </span>
                    </div>

                    <div style={{ fontSize: '12px', fontWeight: 600, color: '#1e293b', marginTop: '2px', marginBottom: '4px' }}>
                      {pt.cameraName || 'Corridor Sensor Node'}
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10.5px', color: '#64748b', borderTop: '1px dashed #e2e8f0', paddingTop: '4px' }}>
                      <span>Type: <strong style={{ color: '#001e40' }}>{pt.vehicleType}</strong></span>
                      <span>ANPR: <strong style={{ color: '#16a34a' }}>{pt.confidence}%</strong></span>
                      {pt.direction && <span>Dir: <strong style={{ color: '#001e40' }}>{pt.direction}</strong></span>}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Pre-Search / Default State: Camera Network Listing */}
          {searchStatus !== 'FOUND' && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              overflowY: 'auto',
              paddingRight: '2px',
              flexGrow: 1
            }}>
              <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>
                Select a target above or enter a registration number to begin corridor investigation.
              </div>
              {cameras.map((cam) => (
                <div
                  key={cam.id}
                  style={{
                    padding: '8px 10px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '6px',
                    backgroundColor: '#ffffff'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                    <strong style={{ fontFamily: 'var(--font-mono)', color: '#001e40', fontSize: '11.5px' }}>
                      {cam.id}
                    </strong>
                    <span style={{
                      fontSize: '9.5px',
                      fontWeight: 700,
                      padding: '1px 6px',
                      borderRadius: '9999px',
                      backgroundColor: cam.status === 'online' ? '#dcfce7' : (cam.status === 'warning' ? '#fef3c7' : '#fee2e2'),
                      color: cam.status === 'online' ? '#166534' : (cam.status === 'warning' ? '#b45309' : '#b91c1c'),
                      textTransform: 'uppercase'
                    }}>
                      {cam.status}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#334155', fontWeight: 600 }}>
                    {cam.name}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', marginTop: '3px' }}>
                    <span>Road: {cam.roadName || cam.road_name || 'N/A'}</span>
                    {cam.speedLimitKmh && <span>Limit: {cam.speedLimitKmh} km/h</span>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
