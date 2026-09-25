import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { initialZones } from '../data/mockData';
import { TrafficMap, CameraMarker, MapLegend } from '../components/gis';
import {
  getCameras,
  createCamera,
  getTrafficEvents,
  getPlateCaptures,
  getGlobalVehicles,
  getVehicleTrajectory,
  getAlerts
} from '../services/gisService';
import { Circle, Popup, Marker, Polyline } from 'react-leaflet';
import L from 'leaflet';

// Location hierarchy data
const MONITORING_REGIONS = [
  {
    state: 'Tamil Nadu',
    cities: [
      { name: 'Coimbatore', center: [11.0183, 76.9580], zoom: 14, configured: true },
      { name: 'Chennai', center: [13.0827, 80.2707], zoom: 12, configured: false },
      { name: 'Madurai', center: [9.9252, 78.1198], zoom: 12, configured: false }
    ]
  }
];

export default function DashboardPage({ onAddNotification, setRoute }) {
  const [mapMode, setMapMode] = useState('radar'); // 'radar' or 'heatmap'
  const [selectedZone, setSelectedZone] = useState(null);

  // Location selector state
  const [selectedState, setSelectedState] = useState('Tamil Nadu');
  const [selectedCity, setSelectedCity] = useState('Coimbatore');

  // Real backend telemetry states
  const [cameras, setCameras] = useState([]);
  const [trafficEvents, setTrafficEvents] = useState({ zones: [], incidents: [] });
  const [anprList, setAnprList] = useState([]);
  const [globalVehicles, setGlobalVehicles] = useState([]);
  const [alerts, setAlerts] = useState([]);

  // Vehicle intelligence states
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [selectedVehicleTrajectory, setSelectedVehicleTrajectory] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  // Map viewport states
  const [mapCenter, setMapCenter] = useState([11.0183, 76.9580]);
  const [mapZoom, setMapZoom] = useState(14);
  const [mapResetKey, setMapResetKey] = useState(0);

  // Add Camera state & Placement mode
  const [isAddCameraMode, setIsAddCameraMode] = useState(false);
  const [temporaryPin, setTemporaryPin] = useState(null);
  const [isSubmittingCamera, setIsSubmittingCamera] = useState(false);
  const [cameraFormError, setCameraFormError] = useState('');
  const [cameraFormSuccess, setCameraFormSuccess] = useState('');
  const [newCameraForm, setNewCameraForm] = useState({
    id: '',
    name: '',
    state: 'Tamil Nadu',
    city: 'Coimbatore',
    road_name: '',
    latitude: '',
    longitude: '',
    speed_limit_kmh: '40',
    status: 'online'
  });

  // Current active city config
  const currentCityConfig = useMemo(() => {
    const st = MONITORING_REGIONS.find((r) => r.state === selectedState);
    if (!st) return null;
    return st.cities.find((c) => c.name === selectedCity) || null;
  }, [selectedState, selectedCity]);

  // Load all initial data from backend APIs
  const fetchAllData = useCallback(() => {
    getCameras().then((cams) => {
      setCameras(cams || []);
      if (cams && cams.length > 0 && selectedCity === 'Coimbatore') {
        const avgLat = cams.reduce((sum, c) => sum + c.latitude, 0) / cams.length;
        const avgLng = cams.reduce((sum, c) => sum + c.longitude, 0) / cams.length;
        setMapCenter([avgLat, avgLng]);
      }
    });

    getTrafficEvents().then((events) => {
      if (events) setTrafficEvents(events);
    });

    getPlateCaptures(15).then((plates) => {
      setAnprList(plates || []);
    });

    getGlobalVehicles(50).then((vList) => {
      setGlobalVehicles(vList || []);
      // Auto-select first vehicle if none selected
      if (vList && vList.length > 0) {
        const first = vList[0];
        setSelectedVehicle(first);
        const queryTerm = first.plate_text || first.global_vehicle_id;
        getVehicleTrajectory(queryTerm).then(setSelectedVehicleTrajectory);
      }
    });

    getAlerts().then((alertItems) => {
      setAlerts(alertItems || []);
    });
  }, [selectedCity]);

  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);

  // City-filtered cameras
  const filteredCameras = useMemo(() => {
    if (!currentCityConfig || !currentCityConfig.configured) {
      return [];
    }
    return cameras.filter((c) => {
      if (c.city) return c.city.toLowerCase() === selectedCity.toLowerCase();
      if (c.locationName && c.locationName.toLowerCase().includes(selectedCity.toLowerCase())) return true;
      return selectedCity === 'Coimbatore';
    });
  }, [cameras, currentCityConfig, selectedCity]);

  // Handle State Change
  const handleStateChange = (newState) => {
    setSelectedState(newState);
    const region = MONITORING_REGIONS.find((r) => r.state === newState);
    if (region && region.cities.length > 0) {
      const firstCity = region.cities[0];
      setSelectedCity(firstCity.name);
      setMapCenter(firstCity.center);
      setMapZoom(firstCity.zoom);
    } else {
      setSelectedCity('');
    }
  };

  // Handle City Change
  const handleCityChange = (newCity) => {
    setSelectedCity(newCity);
    const region = MONITORING_REGIONS.find((r) => r.state === selectedState);
    const city = region?.cities.find((c) => c.name === newCity);
    if (city) {
      setMapCenter(city.center);
      setMapZoom(city.zoom);
    }
  };

  // Handle Reset Map button
  const handleResetMap = () => {
    if (currentCityConfig) {
      setMapCenter(currentCityConfig.center);
      setMapZoom(currentCityConfig.zoom);
    } else {
      setMapCenter([11.0183, 76.9580]);
      setMapZoom(14);
    }
    setMapResetKey((k) => k + 1);
    setSelectedZone(null);
  };

  // Handle Map Click for Camera Placement
  const handleMapClick = (coords) => {
    if (!isAddCameraMode) return;
    const latStr = coords.lat.toFixed(5);
    const lngStr = coords.lng.toFixed(5);
    setTemporaryPin([coords.lat, coords.lng]);
    setNewCameraForm((prev) => ({
      ...prev,
      latitude: latStr,
      longitude: lngStr
    }));
    setCameraFormError('');
  };

  // Handle Add Camera Form Submission
  const handleSaveCamera = async (e) => {
    e?.preventDefault();
    setCameraFormError('');
    setCameraFormSuccess('');

    // Validation
    const id = newCameraForm.id.trim();
    const name = newCameraForm.name.trim();
    const road = newCameraForm.road_name.trim();
    const lat = parseFloat(newCameraForm.latitude);
    const lng = parseFloat(newCameraForm.longitude);
    const speed = parseFloat(newCameraForm.speed_limit_kmh);

    if (!id) {
      setCameraFormError('Camera ID is required (e.g. CAM_05)');
      return;
    }
    if (cameras.some((c) => c.id.toLowerCase() === id.toLowerCase())) {
      setCameraFormError(`Camera ID "${id}" already exists. ID must be unique.`);
      return;
    }
    if (!name) {
      setCameraFormError('Camera Name is required.');
      return;
    }
    if (!road) {
      setCameraFormError('Road Name is required.');
      return;
    }
    if (isNaN(lat) || lat < -90 || lat > 90) {
      setCameraFormError('Valid Latitude (-90 to +90) is required.');
      return;
    }
    if (isNaN(lng) || lng < -180 || lng > 180) {
      setCameraFormError('Valid Longitude (-180 to +180) is required.');
      return;
    }
    if (isNaN(speed) || speed <= 0) {
      setCameraFormError('Speed limit must be a positive numeric value.');
      return;
    }

    setIsSubmittingCamera(true);
    try {
      const payload = {
        id,
        name,
        latitude: lat,
        longitude: lng,
        road_name: road,
        speed_limit_kmh: speed,
        status: newCameraForm.status,
        city: selectedCity,
        state: selectedState,
        location_name: `${road}, ${selectedCity}, ${selectedState}`
      };

      const created = await createCamera(payload);
      setCameraFormSuccess(`Camera ${created.id} saved to PostgreSQL/PostGIS!`);
      if (onAddNotification) {
        onAddNotification(`Sensor node ${created.id} registered with PostGIS coordinates.`);
      }

      // Refresh camera list
      const updatedCams = await getCameras();
      setCameras(updatedCams || []);

      // Reset form and close mode
      setTimeout(() => {
        setIsAddCameraMode(false);
        setTemporaryPin(null);
        setCameraFormSuccess('');
        setNewCameraForm({
          id: '',
          name: '',
          state: selectedState,
          city: selectedCity,
          road_name: '',
          latitude: '',
          longitude: '',
          speed_limit_kmh: '40',
          status: 'online'
        });
      }, 1200);
    } catch (err) {
      setCameraFormError(err.message || 'Failed to save camera to database.');
    } finally {
      setIsSubmittingCamera(false);
    }
  };

  // Handle Vehicle Selection
  const handleSelectVehicle = (v) => {
    setSelectedVehicle(v);
    const queryTerm = v.plate_text || v.global_vehicle_id;
    getVehicleTrajectory(queryTerm).then((traj) => {
      setSelectedVehicleTrajectory(traj);
      if (traj && traj.detections && traj.detections.length > 0) {
        const coords = traj.detections.map((d) => [d.latitude, d.longitude]);
        setMapCenter(coords[0]);
      }
    });
  };

  // Handle Vehicle Search
  const handleSearchVehicle = async (e) => {
    e?.preventDefault();
    if (!searchQuery.trim()) {
      fetchAllData();
      return;
    }
    setIsSearching(true);
    try {
      const results = await getGlobalVehicles(20, searchQuery.trim());
      setGlobalVehicles(results);
      if (results && results.length > 0) {
        handleSelectVehicle(results[0]);
      } else {
        // Also attempt direct trajectory lookup
        const traj = await getVehicleTrajectory(searchQuery.trim());
        if (traj && traj.detections && traj.detections.length > 0) {
          const synthVehicle = {
            global_vehicle_id: traj.plate.startsWith('GV_') ? traj.plate : `GV_${traj.plate}`,
            vehicle_class: traj.detections[0]?.vehicleType || 'Vehicle',
            plate_text: traj.plate,
            match_confidence: 0.98,
            camera_observations: traj.detections.map((d) => d.cameraId),
            first_seen_cam: traj.detections[0]?.cameraId,
            last_seen_cam: traj.detections[traj.detections.length - 1]?.cameraId,
            camera_count: traj.detections.length,
            timeline: traj.detections.map((d, i) => ({
              camera_id: d.cameraId,
              camera_name: d.cameraName || d.cameraId,
              road_name: d.cameraName || 'Corridor Node',
              timestamp: d.timestamp,
              status: i === 0 ? 'First Detection' : i === traj.detections.length - 1 ? 'Last Seen' : 'Detected',
              confidence: d.confidence
            }))
          };
          setSelectedVehicle(synthVehicle);
          setSelectedVehicleTrajectory(traj);
        }
      }
    } finally {
      setIsSearching(false);
    }
  };

  // Observed camera IDs for map highlighting
  const observedCameraIdSet = useMemo(() => {
    if (!selectedVehicle || !selectedVehicle.camera_observations) return new Set();
    return new Set(selectedVehicle.camera_observations);
  }, [selectedVehicle]);

  // Observed camera sequence ordered array
  const observedCameraSequence = useMemo(() => {
    if (!selectedVehicle || !selectedVehicle.camera_observations) return [];
    return selectedVehicle.camera_observations;
  }, [selectedVehicle]);

  // Trajectory polyline coordinates
  const trajectoryPolyline = useMemo(() => {
    if (selectedVehicleTrajectory && selectedVehicleTrajectory.detections && selectedVehicleTrajectory.detections.length > 0) {
      return selectedVehicleTrajectory.detections.map((d) => [d.latitude, d.longitude]);
    }
    // Fallback: connect observed cameras from camera network
    if (observedCameraSequence.length > 1) {
      const coords = [];
      for (const cid of observedCameraSequence) {
        const cam = cameras.find((c) => c.id === cid);
        if (cam) coords.push([cam.latitude, cam.longitude]);
      }
      return coords;
    }
    return [];
  }, [selectedVehicleTrajectory, observedCameraSequence, cameras]);

  const incidentIcon = useMemo(() => L.divIcon({
    html: `
      <div style="width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; background-color: #dc2626; border: 2px solid #ffffff; border-radius: 9999px; box-shadow: 0 0 10px rgba(220, 38, 38, 0.7); cursor: pointer;">
        <span style="color: #ffffff; font-size: 12px; font-weight: 900; line-height: 1;">!</span>
      </div>
    `,
    className: 'gis-incident-pin',
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -14]
  }), []);

  const tempPlacementIcon = useMemo(() => L.divIcon({
    html: `
      <div style="position: relative; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center;">
        <span style="position: absolute; inset: -6px; border-radius: 9999px; background: rgba(2, 132, 199, 0.4); animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;"></span>
        <div style="width: 28px; height: 28px; border-radius: 9999px; background-color: #0284c7; border: 3px solid #ffffff; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.6); display: flex; align-items: center; justify-content: center; color: #ffffff; font-weight: 900; font-size: 14px;">
          +
        </div>
      </div>
    `,
    className: 'temp-placement-pin',
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -20]
  }), []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', boxSizing: 'border-box' }}>
      {/* 1. TOP SUMMARY METRICS ROW (Authentic Real Database Telemetry) */}
      <section className="container-7xl" style={{ paddingTop: '1.75rem', paddingBottom: '1.25rem' }}>
        <div className="dashboard-kpi-grid">
          {/* Card 1: Configured Cameras */}
          <div style={{
            backgroundColor: '#ffffff',
            padding: '1.25rem',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)',
            border: '1px solid #e2e8f0',
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            height: '100%',
            boxSizing: 'border-box'
          }}>
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '3px',
              background: 'linear-gradient(90deg, #38bdf8 0%, #003366 100%)'
            }} />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#64748b' }}>
                  Configured Cameras
                </span>
                <span style={{
                  padding: '2px 8px',
                  backgroundColor: '#ecfdf5',
                  color: '#065f46',
                  border: '1px solid #a7f3d0',
                  borderRadius: '9999px',
                  fontSize: '11px',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px'
                }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#059669' }} className="animate-pulse-subtle" />
                  <span>PostgreSQL Active</span>
                </span>
              </div>
              <div style={{ fontSize: '32px', fontWeight: 800, color: 'var(--primary-container, #003366)', letterSpacing: '-0.03em', display: 'flex', alignItems: 'baseline', gap: '4px' }}>
                {cameras.length}
                <span style={{ fontSize: '13px', fontWeight: 500, color: '#64748b' }}>sensor nodes</span>
              </div>
            </div>
            <div style={{
              marginTop: '1rem',
              paddingTop: '0.75rem',
              borderTop: '1px solid #f1f5f9',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#64748b'
            }}>
              <span>City: {selectedCity}</span>
              <span style={{ color: '#003366', fontWeight: 700 }}>
                {filteredCameras.length} active in view
              </span>
            </div>
          </div>

          {/* Card 2: Vehicles Detected */}
          <div style={{
            backgroundColor: '#ffffff',
            padding: '1.25rem',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)',
            border: '1px solid #e2e8f0',
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            height: '100%',
            boxSizing: 'border-box'
          }}>
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '3px',
              background: 'linear-gradient(90deg, #10b981 0%, #047857 100%)'
            }} />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#64748b' }}>
                  Vehicles Detected
                </span>
                <span style={{
                  padding: '2px 8px',
                  backgroundColor: '#f0fdf4',
                  color: '#15803d',
                  border: '1px solid #bbf7d0',
                  borderRadius: '9999px',
                  fontSize: '11px',
                  fontWeight: 700
                }}>
                  Canonical Entities
                </span>
              </div>
              <div style={{ fontSize: '32px', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.03em', display: 'flex', alignItems: 'baseline', gap: '4px' }}>
                {globalVehicles.length}
                <span style={{ fontSize: '13px', fontWeight: 500, color: '#64748b' }}>global IDs</span>
              </div>
            </div>
            <div style={{
              marginTop: '1rem',
              paddingTop: '0.75rem',
              borderTop: '1px solid #f1f5f9',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#64748b'
            }}>
              <span>Multi-Camera Grid</span>
              <span style={{ color: '#047857', fontWeight: 700 }}>AI Re-ID Indexed</span>
            </div>
          </div>

          {/* Card 3: Live ANPR Captures */}
          <div style={{
            backgroundColor: '#ffffff',
            padding: '1.25rem',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)',
            border: '1px solid #e2e8f0',
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            height: '100%',
            boxSizing: 'border-box'
          }}>
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '3px',
              background: 'linear-gradient(90deg, #6366f1 0%, #4338ca 100%)'
            }} />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#64748b' }}>
                  ANPR Captures
                </span>
                <span style={{
                  padding: '2px 8px',
                  backgroundColor: '#e0e7ff',
                  color: '#4338ca',
                  border: '1px solid #c7d2fe',
                  borderRadius: '9999px',
                  fontSize: '11px',
                  fontWeight: 700
                }}>
                  PaddleOCR Real-time
                </span>
              </div>
              <div style={{ fontSize: '32px', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.03em', display: 'flex', alignItems: 'baseline', gap: '4px' }}>
                {anprList.length}
                <span style={{ fontSize: '13px', fontWeight: 500, color: '#64748b' }}>verified</span>
              </div>
            </div>
            <div style={{
              marginTop: '1rem',
              paddingTop: '0.75rem',
              borderTop: '1px solid #f1f5f9',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#64748b'
            }}>
              <span>Status: Operational</span>
              <span style={{ color: '#4338ca', fontWeight: 700 }}>Database Sighted</span>
            </div>
          </div>

          {/* Card 4: Active Enforcement Alerts */}
          <div style={{
            backgroundColor: '#ffffff',
            padding: '1.25rem',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)',
            border: '1px solid #e2e8f0',
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            height: '100%',
            boxSizing: 'border-box'
          }}>
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '3px',
              background: 'linear-gradient(90deg, #f59e0b 0%, #dc2626 100%)'
            }} />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#64748b' }}>
                  Active Alerts
                </span>
                <span style={{
                  padding: '2px 8px',
                  backgroundColor: '#fef2f2',
                  color: '#991b1b',
                  border: '1px solid #fecaca',
                  borderRadius: '9999px',
                  fontSize: '11px',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px'
                }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#dc2626' }} className="animate-pulse-subtle" />
                  <span>Enforcement</span>
                </span>
              </div>
              <div style={{ fontSize: '32px', fontWeight: 800, color: '#dc2626', letterSpacing: '-0.03em' }}>
                {alerts.length}
              </div>
            </div>
            <div style={{
              marginTop: '1rem',
              paddingTop: '0.75rem',
              borderTop: '1px solid #f1f5f9',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#64748b'
            }}>
              <span>Pending Review</span>
              <button
                type="button"
                onClick={() => setRoute && setRoute('/tracking')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--primary-container, #003366)',
                  fontWeight: 700,
                  cursor: 'pointer',
                  fontSize: '12px',
                  padding: 0
                }}
              >
                View Incident Tracking →
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* 2. GIS TRAFFIC MONITORING & LIVE ANPR SECTION */}
      <section className="container-7xl" style={{ paddingBottom: '1.75rem' }}>
        {/* GIS LOCATION SELECTION & ACTION BAR */}
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          padding: '1rem 1.25rem',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          boxShadow: '0 2px 6px rgba(0, 35, 80, 0.03)'
        }}>
          {/* State and City Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '20px', color: '#003366' }}>location_on</span>
              <span style={{ fontSize: '12px', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#003366' }}>
                Traffic Monitoring Location:
              </span>
            </div>

            {/* State Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label htmlFor="state-select" style={{ fontSize: '12px', fontWeight: 600, color: '#475569' }}>State:</label>
              <select
                id="state-select"
                value={selectedState}
                onChange={(e) => handleStateChange(e.target.value)}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  border: '1px solid #cbd5e1',
                  backgroundColor: '#f8fafc',
                  fontSize: '12.5px',
                  fontWeight: 700,
                  color: '#003366',
                  cursor: 'pointer'
                }}
              >
                {MONITORING_REGIONS.map((r) => (
                  <option key={r.state} value={r.state}>{r.state}</option>
                ))}
              </select>
            </div>

            {/* City Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label htmlFor="city-select" style={{ fontSize: '12px', fontWeight: 600, color: '#475569' }}>City:</label>
              <select
                id="city-select"
                value={selectedCity}
                onChange={(e) => handleCityChange(e.target.value)}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  border: '1px solid #cbd5e1',
                  backgroundColor: '#f8fafc',
                  fontSize: '12.5px',
                  fontWeight: 700,
                  color: '#003366',
                  cursor: 'pointer'
                }}
              >
                {MONITORING_REGIONS.find((r) => r.state === selectedState)?.cities.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name} {c.configured ? '(Configured)' : '(No Network)'}
                  </option>
                ))}
              </select>
            </div>

            {/* Selected Location Indicator Badge */}
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              backgroundColor: currentCityConfig?.configured ? '#eff6ff' : '#fef2f2',
              color: currentCityConfig?.configured ? '#003366' : '#991b1b',
              border: `1px solid ${currentCityConfig?.configured ? '#bfdbfe' : '#fecaca'}`,
              borderRadius: '9999px',
              fontSize: '11.5px',
              fontWeight: 700
            }}>
              <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: currentCityConfig?.configured ? '#0284c7' : '#dc2626' }} />
              <span>{selectedCity}, {selectedState} • {filteredCameras.length} Cameras</span>
            </span>
          </div>

          {/* Action Buttons: Add Camera & Reset Map */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              type="button"
              onClick={() => {
                setIsAddCameraMode((prev) => !prev);
                setCameraFormError('');
                setCameraFormSuccess('');
              }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 14px',
                borderRadius: '6px',
                border: isAddCameraMode ? '2px solid #0284c7' : '1px solid #003366',
                backgroundColor: isAddCameraMode ? '#e0f2fe' : '#003366',
                color: isAddCameraMode ? '#003366' : '#ffffff',
                fontSize: '12.5px',
                fontWeight: 700,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                {isAddCameraMode ? 'close' : 'add_location_alt'}
              </span>
              <span>{isAddCameraMode ? 'Cancel Placement' : '+ Add Camera'}</span>
            </button>

            <button
              type="button"
              onClick={handleResetMap}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                backgroundColor: '#ffffff',
                color: '#334155',
                fontSize: '12.5px',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>my_location</span>
              <span>Reset Map</span>
            </button>
          </div>
        </div>

        {/* CAMERA PLACEMENT BANNER / INTERACTIVE FORM OVERLAY */}
        {isAddCameraMode && (
          <div style={{
            backgroundColor: '#f0f9ff',
            border: '2px solid #0284c7',
            borderRadius: '12px',
            padding: '1.25rem',
            marginBottom: '1rem',
            boxShadow: '0 4px 14px rgba(2, 132, 199, 0.15)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '22px', color: '#0284c7' }}>pin_drop</span>
                <h3 style={{ fontSize: '14px', fontWeight: 800, color: '#003366', margin: 0 }}>
                  Camera Placement Mode Active
                </h3>
                <span style={{ fontSize: '12px', color: '#475569' }}>
                  (Click any point on the Leaflet map below to set GPS coordinates, or type them manually)
                </span>
              </div>
              <button
                type="button"
                onClick={() => setIsAddCameraMode(false)}
                style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: '18px', fontWeight: 800 }}
              >
                ✕
              </button>
            </div>

            {cameraFormError && (
              <div style={{ backgroundColor: '#fee2e2', border: '1px solid #fecaca', color: '#991b1b', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', marginBottom: '0.75rem', fontWeight: 600 }}>
                ⚠️ {cameraFormError}
              </div>
            )}
            {cameraFormSuccess && (
              <div style={{ backgroundColor: '#ecfdf5', border: '1px solid #a7f3d0', color: '#065f46', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', marginBottom: '0.75rem', fontWeight: 700 }}>
                ✓ {cameraFormSuccess}
              </div>
            )}

            <form onSubmit={handleSaveCamera} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px', alignItems: 'end' }}>
              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#475569', marginBottom: '3px' }}>Camera ID *</label>
                <input
                  type="text"
                  placeholder="e.g. CAM_05"
                  value={newCameraForm.id}
                  onChange={(e) => setNewCameraForm({ ...newCameraForm, id: e.target.value })}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '12px', boxSizing: 'border-box' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#475569', marginBottom: '3px' }}>Camera Name *</label>
                <input
                  type="text"
                  placeholder="e.g. Avinashi Road Cam"
                  value={newCameraForm.name}
                  onChange={(e) => setNewCameraForm({ ...newCameraForm, name: e.target.value })}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '12px', boxSizing: 'border-box' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#475569', marginBottom: '3px' }}>Road Name *</label>
                <input
                  type="text"
                  placeholder="e.g. Avinashi Road"
                  value={newCameraForm.road_name}
                  onChange={(e) => setNewCameraForm({ ...newCameraForm, road_name: e.target.value })}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '12px', boxSizing: 'border-box' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#475569', marginBottom: '3px' }}>Latitude *</label>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 11.0250"
                  value={newCameraForm.latitude}
                  onChange={(e) => setNewCameraForm({ ...newCameraForm, latitude: e.target.value })}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '12px', boxSizing: 'border-box' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#475569', marginBottom: '3px' }}>Longitude *</label>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 76.9630"
                  value={newCameraForm.longitude}
                  onChange={(e) => setNewCameraForm({ ...newCameraForm, longitude: e.target.value })}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '12px', boxSizing: 'border-box' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#475569', marginBottom: '3px' }}>Speed Limit (km/h)</label>
                <input
                  type="number"
                  placeholder="40"
                  value={newCameraForm.speed_limit_kmh}
                  onChange={(e) => setNewCameraForm({ ...newCameraForm, speed_limit_kmh: e.target.value })}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '12px', boxSizing: 'border-box' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#475569', marginBottom: '3px' }}>Status</label>
                <select
                  value={newCameraForm.status}
                  onChange={(e) => setNewCameraForm({ ...newCameraForm, status: e.target.value })}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '12px', boxSizing: 'border-box' }}
                >
                  <option value="online">Online</option>
                  <option value="warning">Degraded</option>
                  <option value="offline">Offline</option>
                </select>
              </div>

              <div>
                <button
                  type="submit"
                  disabled={isSubmittingCamera}
                  style={{
                    width: '100%',
                    padding: '7px 14px',
                    borderRadius: '6px',
                    backgroundColor: '#0284c7',
                    color: '#ffffff',
                    border: 'none',
                    fontWeight: 700,
                    fontSize: '12px',
                    cursor: isSubmittingCamera ? 'wait' : 'pointer'
                  }}
                >
                  {isSubmittingCamera ? 'Persisting...' : 'Save Camera'}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* 2 COLUMNS: LEFT GIS MAP (1.4fr), RIGHT REAL ANPR FEED (1fr) */}
        <div className="dashboard-main-grid">
          {/* Left: Interactive Map */}
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '16px',
            border: '1px solid #e2e8f0',
            boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between'
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <h2 style={{ fontSize: '17px', fontWeight: 800, color: 'var(--primary-container, #003366)' }}>
                      Regional Traffic Density &amp; Surveillance
                    </h2>
                    <span style={{ width: '8px', height: '8px', borderRadius: '9999px', backgroundColor: '#0284c7' }} className="animate-pulse-subtle" />
                  </div>
                  <p style={{ fontSize: '12px', color: '#64748b' }}>
                    {selectedCity} Traffic Intelligence Network — {selectedState}
                  </p>
                </div>

                {/* Radar vs Heatmap controls */}
                <div style={{
                  display: 'flex',
                  gap: '4px',
                  backgroundColor: '#f1f5f9',
                  padding: '4px',
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0'
                }}>
                  <button
                    type="button"
                    onClick={() => setMapMode('radar')}
                    style={{
                      padding: '4px 10px',
                      borderRadius: '6px',
                      border: 'none',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      backgroundColor: mapMode === 'radar' ? '#ffffff' : 'transparent',
                      color: mapMode === 'radar' ? '#003366' : '#64748b',
                      boxShadow: mapMode === 'radar' ? '0 1px 3px rgba(0, 0, 0, 0.1)' : 'none'
                    }}
                  >
                    Live Radar
                  </button>
                  <button
                    type="button"
                    onClick={() => setMapMode('heatmap')}
                    style={{
                      padding: '4px 10px',
                      borderRadius: '6px',
                      border: 'none',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      backgroundColor: mapMode === 'heatmap' ? '#003366' : 'transparent',
                      color: mapMode === 'heatmap' ? '#ffffff' : '#64748b',
                      boxShadow: mapMode === 'heatmap' ? '0 1px 3px rgba(0, 0, 0, 0.1)' : 'none'
                    }}
                  >
                    Heatmap
                  </button>
                </div>
              </div>

              {/* Map Viewport */}
              <div style={{
                position: 'relative',
                width: '100%',
                height: '420px',
                borderRadius: '10px',
                overflow: 'hidden',
                backgroundColor: '#0f172a',
                border: '1px solid #cbd5e1',
                marginBottom: '1rem'
              }}>
                <div className="tricorn-ribbon" style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', zIndex: 1001, pointerEvents: 'none' }} />

                <TrafficMap
                  key={`${selectedCity}-${mapResetKey}`}
                  center={mapCenter}
                  zoom={mapZoom}
                  height="100%"
                  showResetButton={false}
                  tileLayerType={mapMode === 'heatmap' ? 'dark' : 'standard'}
                  onMapClick={handleMapClick}
                  autoFitCoords={filteredCameras.length > 0 ? filteredCameras.map((c) => [c.latitude, c.longitude]) : undefined}
                >
                  {/* Traffic Zones */}
                  {trafficEvents.zones.map((zone) => (
                    <Circle
                      key={zone.id}
                      center={zone.center}
                      radius={zone.radiusMeters}
                      pathOptions={{
                        color: zone.color,
                        fillColor: zone.color,
                        fillOpacity: mapMode === 'heatmap' ? 0.35 : 0.15,
                        weight: 2
                      }}
                    >
                      <Popup>
                        <div style={{ padding: '6px 8px', fontSize: '11.5px', fontFamily: 'var(--font-heading, sans-serif)' }}>
                          <strong style={{ color: '#003366', display: 'block', fontSize: '12px' }}>{zone.name}</strong>
                          <span style={{ color: zone.color, fontWeight: 700 }}>{zone.statusText}</span>
                          <div style={{ marginTop: '4px', color: '#64748b', fontSize: '10.5px' }}>
                            Density: <strong>{zone.density}</strong> • Speed: <strong>{zone.avgSpeed}</strong>
                          </div>
                        </div>
                      </Popup>
                    </Circle>
                  ))}

                  {/* Incidents */}
                  {trafficEvents.incidents.map((inc) => (
                    <Marker
                      key={inc.id}
                      position={[inc.latitude, inc.longitude]}
                      icon={incidentIcon}
                    >
                      <Popup>
                        <div style={{ padding: '8px 10px', fontSize: '11.5px', fontFamily: 'var(--font-heading, sans-serif)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', marginBottom: '4px' }}>
                            <strong style={{ color: '#dc2626' }}>{inc.id}</strong>
                            <span style={{ backgroundColor: '#fee2e2', color: '#991b1b', fontSize: '10px', padding: '1px 6px', borderRadius: '4px', fontWeight: 700, textTransform: 'uppercase' }}>{inc.severity}</span>
                          </div>
                          <div style={{ fontWeight: 700, color: '#0f172a', marginBottom: '2px' }}>{inc.title}</div>
                          <div style={{ color: '#64748b', fontSize: '11px', marginBottom: '4px' }}>{inc.corridor} • {inc.timestamp}</div>
                          <p style={{ color: '#334155', fontSize: '10.5px', margin: 0 }}>{inc.description}</p>
                        </div>
                      </Popup>
                    </Marker>
                  ))}

                  {/* Camera Markers with Highlight support */}
                  {filteredCameras.map((cam) => {
                    const isObserved = observedCameraIdSet.has(cam.id);
                    const seqIdx = isObserved ? observedCameraSequence.indexOf(cam.id) : undefined;
                    return (
                      <CameraMarker
                        key={cam.id}
                        camera={cam}
                        isHighlighted={isObserved}
                        sequenceIndex={seqIdx}
                      />
                    );
                  })}

                  {/* Temporary Placement Marker */}
                  {temporaryPin && (
                    <Marker position={temporaryPin} icon={tempPlacementIcon}>
                      <Popup>
                        <div style={{ padding: '6px 8px', fontSize: '11px' }}>
                          <strong>Target Placement</strong><br />
                          GPS: {temporaryPin[0].toFixed(5)}°N, {temporaryPin[1].toFixed(5)}°E
                        </div>
                      </Popup>
                    </Marker>
                  )}

                  {/* Selected Vehicle Trajectory Polyline */}
                  {trajectoryPolyline.length > 1 && (
                    <Polyline
                      positions={trajectoryPolyline}
                      pathOptions={{
                        color: '#0284c7',
                        weight: 4,
                        opacity: 0.85,
                        dashArray: '6, 8',
                        lineJoin: 'round'
                      }}
                    />
                  )}

                  <MapLegend position="bottomleft" />
                </TrafficMap>

                {/* Empty State Overlay for Unconfigured Locations */}
                {filteredCameras.length === 0 && (
                  <div style={{
                    position: 'absolute',
                    inset: 0,
                    backgroundColor: 'rgba(15, 23, 42, 0.82)',
                    backdropFilter: 'blur(4px)',
                    zIndex: 1000,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#ffffff',
                    padding: '1.5rem',
                    textAlign: 'center'
                  }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '42px', color: '#f59e0b', marginBottom: '8px' }}>
                      videocam_off
                    </span>
                    <h3 style={{ fontSize: '16px', fontWeight: 800, margin: '0 0 6px 0', color: '#ffffff' }}>
                      No NETRA camera network is currently configured for this location.
                    </h3>
                    <p style={{ fontSize: '12.5px', color: '#94a3b8', maxWidth: '420px', margin: 0 }}>
                      The active sensor network for the current deployment is established in <strong>Coimbatore, Tamil Nadu</strong>.
                    </p>
                    <button
                      type="button"
                      onClick={() => handleCityChange('Coimbatore')}
                      style={{
                        marginTop: '12px',
                        padding: '6px 14px',
                        borderRadius: '6px',
                        backgroundColor: '#0284c7',
                        color: '#ffffff',
                        border: 'none',
                        fontSize: '12px',
                        fontWeight: 700,
                        cursor: 'pointer'
                      }}
                    >
                      Switch to Coimbatore Grid
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Map footer status bar */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#64748b',
              paddingTop: '0.75rem',
              borderTop: '1px solid #f1f5f9'
            }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '16px', color: '#0284c7' }}>map</span>
                <span>Active City Bounds: {selectedCity} ({filteredCameras.length} nodes)</span>
              </span>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                fontWeight: 600,
                color: 'var(--primary-container, #003366)',
                backgroundColor: '#f0f9ff',
                padding: '2px 8px',
                borderRadius: '4px',
                border: '1px solid rgba(186, 230, 253, 0.7)'
              }}>
                PostGIS SRID 4326
              </span>
            </div>
          </div>

          {/* Right: Live ANPR Feed (Real Backend Sighted Stream) */}
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '16px',
            border: '1px solid #e2e8f0',
            boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between'
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <div>
                  <h2 style={{ fontSize: '17px', fontWeight: 800, color: 'var(--primary-container, #003366)' }}>
                    Live ANPR Feed
                  </h2>
                  <p style={{ fontSize: '12px', color: '#64748b' }}>Automatic Number Plate Recognition Stream</p>
                </div>
                <span style={{
                  padding: '2px 8px',
                  backgroundColor: '#f0fdfa',
                  color: '#0f766e',
                  border: '1px solid #99f6e4',
                  borderRadius: '9999px',
                  fontSize: '11px',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px'
                }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#0d9488' }} className="animate-pulse-subtle" />
                  <span>Real Telemetry</span>
                </span>
              </div>

              {/* Table of Real Captures */}
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', fontSize: '12.5px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', backgroundColor: '#f8fafc', color: '#64748b', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      <th style={{ padding: '8px 10px', fontWeight: 700 }}>Plate / Type</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700 }}>Cam ID</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700 }}>Timestamp</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700, textAlign: 'right' }}>Conf.</th>
                    </tr>
                  </thead>
                  <tbody style={{ color: '#0f172a' }}>
                    {anprList.length > 0 ? (
                      anprList.slice(0, 7).map((item) => (
                        <tr
                          key={item.id}
                          style={{ borderBottom: '1px solid #f1f5f9', cursor: 'pointer', transition: 'background-color 0.1s' }}
                          onClick={() => {
                            setSearchQuery(item.plate);
                            getGlobalVehicles(10, item.plate).then((res) => {
                              if (res && res.length > 0) handleSelectVehicle(res[0]);
                            });
                          }}
                          onMouseOver={(e) => { e.currentTarget.style.backgroundColor = '#f0f9ff'; }}
                          onMouseOut={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
                        >
                          <td style={{ padding: '8px 10px' }}>
                            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--primary-container, #003366)', fontSize: '13px' }}>
                              {item.plate}
                            </div>
                            <div style={{ fontSize: '11px', color: '#64748b' }}>{item.type}</div>
                          </td>
                          <td style={{ padding: '8px 8px', color: '#64748b', fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>
                            {item.camId}
                          </td>
                          <td style={{ padding: '8px 8px', color: '#64748b', fontSize: '11.5px' }}>
                            {item.timestamp}
                          </td>
                          <td style={{ padding: '8px 8px', textAlign: 'right' }}>
                            <span style={{
                              padding: '2px 6px',
                              backgroundColor: parseFloat(item.confidence) > 90 ? '#ecfdf5' : '#fef3c7',
                              color: parseFloat(item.confidence) > 90 ? '#065f46' : '#92400e',
                              border: `1px solid ${parseFloat(item.confidence) > 90 ? '#a7f3d0' : '#fde68a'}`,
                              borderRadius: '4px',
                              fontSize: '11px',
                              fontWeight: 700
                            }}>
                              {item.confidence}
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={4} style={{ padding: '1.5rem', textAlign: 'center', color: '#64748b' }}>
                          No recent ANPR captures found in database.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              color: '#64748b',
              paddingTop: '0.75rem',
              borderTop: '1px solid #f1f5f9'
            }}>
              <span>Showing real database sightings</span>
              <button
                type="button"
                onClick={() => setRoute && setRoute('/anpr')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--primary-container, #003366)',
                  fontWeight: 700,
                  cursor: 'pointer',
                  fontSize: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: 0
                }}
              >
                <span>Full ANPR Stream</span>
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>arrow_forward</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* 3. NEW VEHICLE INTELLIGENCE & MULTI-CAMERA TRACKING SECTION (Replaces System Health & Quick Actions) */}
      <section className="container-7xl" style={{ paddingBottom: '2.5rem' }}>
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          border: '1px solid #e2e8f0',
          boxShadow: '0 4px 16px rgba(0, 35, 80, 0.06)',
          padding: '1.75rem',
          position: 'relative'
        }}>
          {/* Header Bar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '1.25rem',
            flexWrap: 'wrap',
            gap: '1rem',
            borderBottom: '1px solid #f1f5f9',
            paddingBottom: '1rem'
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '24px', color: '#003366' }}>directions_car</span>
                <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#003366', margin: 0 }}>
                  Vehicle Tracking Intelligence
                </h2>
                <span style={{
                  fontSize: '11px',
                  fontWeight: 700,
                  padding: '2px 8px',
                  backgroundColor: '#eff6ff',
                  color: '#003366',
                  borderRadius: '9999px',
                  border: '1px solid #bfdbfe'
                }}>
                  Cross-Camera Re-ID
                </span>
              </div>
              <p style={{ fontSize: '12px', color: '#64748b', margin: '4px 0 0 0' }}>
                Multi-camera spatial correlation, trajectory reconstruction, and detection timeline across sensor corridors.
              </p>
            </div>

            {/* Vehicle Search Box */}
            <form onSubmit={handleSearchVehicle} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ position: 'relative' }}>
                <input
                  type="text"
                  placeholder="Search Vehicle / Plate (e.g. GV_000001, TN07CY2784)"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    padding: '7px 12px 7px 32px',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '12.5px',
                    width: '320px',
                    fontFamily: 'var(--font-mono, monospace)',
                    outline: 'none'
                  }}
                />
                <span className="material-symbols-outlined" style={{ position: 'absolute', left: '8px', top: '7px', fontSize: '18px', color: '#94a3b8' }}>
                  search
                </span>
              </div>
              <button
                type="submit"
                disabled={isSearching}
                style={{
                  padding: '7px 14px',
                  borderRadius: '8px',
                  backgroundColor: '#003366',
                  color: '#ffffff',
                  border: 'none',
                  fontSize: '12px',
                  fontWeight: 700,
                  cursor: isSearching ? 'wait' : 'pointer'
                }}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </button>
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery('');
                    fetchAllData();
                  }}
                  style={{
                    padding: '7px 10px',
                    borderRadius: '8px',
                    backgroundColor: '#f1f5f9',
                    color: '#64748b',
                    border: '1px solid #cbd5e1',
                    fontSize: '12px',
                    cursor: 'pointer'
                  }}
                >
                  Clear
                </button>
              )}
            </form>
          </div>

          {/* MAIN CONTENT: 2 COLUMNS (Left: Quick Selection Table, Right: Selected Vehicle Info & Timeline) */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '1.5rem' }}>
            {/* LEFT: Quick Vehicle Selection Table */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <h4 style={{ fontSize: '13px', fontWeight: 800, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>
                  Recent Global Vehicles ({globalVehicles.length})
                </h4>
                <span style={{ fontSize: '11px', color: '#64748b' }}>Select a row to inspect</span>
              </div>

              <div style={{ overflowX: 'auto', border: '1px solid #e2e8f0', borderRadius: '8px', maxHeight: '420px', overflowY: 'auto' }}>
                <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', fontSize: '12px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f8fafc', color: '#64748b', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', position: 'sticky', top: 0, zIndex: 5, borderBottom: '1px solid #e2e8f0' }}>
                      <th style={{ padding: '8px 10px', fontWeight: 700 }}>Global ID</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700 }}>Plate</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700 }}>Class</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700, textAlign: 'center' }}>Cameras</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700 }}>Last Seen</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700, textAlign: 'right' }}>Conf.</th>
                      <th style={{ padding: '8px 8px', fontWeight: 700, textAlign: 'center' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody style={{ color: '#0f172a' }}>
                    {globalVehicles.map((v) => {
                      const isSelected = selectedVehicle?.global_vehicle_id === v.global_vehicle_id;
                      const confPct = v.match_confidence
                        ? (v.match_confidence > 1 ? `${v.match_confidence.toFixed(0)}%` : `${(v.match_confidence * 100).toFixed(0)}%`)
                        : 'N/A';
                      const plateDisplay = v.plate_text || 'Plate N/A';

                      return (
                        <tr
                          key={v.global_vehicle_id}
                          onClick={() => handleSelectVehicle(v)}
                          style={{
                            borderBottom: '1px solid #f1f5f9',
                            backgroundColor: isSelected ? '#eff6ff' : 'transparent',
                            cursor: 'pointer',
                            transition: 'background-color 0.1s'
                          }}
                          onMouseOver={(e) => { if (!isSelected) e.currentTarget.style.backgroundColor = '#f8fafc'; }}
                          onMouseOut={(e) => { if (!isSelected) e.currentTarget.style.backgroundColor = 'transparent'; }}
                        >
                          <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono, monospace)', fontWeight: 700, color: '#003366' }}>
                            {v.global_vehicle_id}
                          </td>
                          <td style={{ padding: '8px 8px', fontFamily: 'var(--font-mono, monospace)', fontWeight: 600 }}>
                            {plateDisplay}
                          </td>
                          <td style={{ padding: '8px 8px', textTransform: 'capitalize', color: '#475569' }}>
                            {v.vehicle_class}
                          </td>
                          <td style={{ padding: '8px 8px', textAlign: 'center' }}>
                            <span style={{
                              padding: '2px 6px',
                              backgroundColor: v.camera_count > 1 ? '#e0f2fe' : '#f1f5f9',
                              color: v.camera_count > 1 ? '#0369a1' : '#64748b',
                              borderRadius: '4px',
                              fontWeight: 700,
                              fontSize: '11px'
                            }}>
                              {v.camera_count || v.camera_observations?.length || 1}
                            </span>
                          </td>
                          <td style={{ padding: '8px 8px', color: '#64748b', fontFamily: 'var(--font-mono, monospace)', fontSize: '11px' }}>
                            {v.last_seen_cam || 'N/A'}
                          </td>
                          <td style={{ padding: '8px 8px', textAlign: 'right', fontWeight: 600, color: '#0f766e' }}>
                            {confPct}
                          </td>
                          <td style={{ padding: '8px 8px', textAlign: 'center' }}>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSelectVehicle(v);
                              }}
                              style={{
                                padding: '3px 8px',
                                borderRadius: '4px',
                                backgroundColor: isSelected ? '#003366' : '#f1f5f9',
                                color: isSelected ? '#ffffff' : '#003366',
                                border: '1px solid #cbd5e1',
                                fontSize: '10.5px',
                                fontWeight: 700,
                                cursor: 'pointer'
                              }}
                            >
                              {isSelected ? 'Active' : 'Select'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* RIGHT: Selected Vehicle Information Card + Camera Sequence + Timeline */}
            <div>
              {selectedVehicle ? (
                <div>
                  {/* Selected Vehicle Info Card */}
                  <div style={{
                    backgroundColor: '#f8fafc',
                    borderRadius: '12px',
                    border: '1px solid #cbd5e1',
                    padding: '1.25rem',
                    marginBottom: '1rem',
                    position: 'relative'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                      <span style={{ fontSize: '11px', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#64748b' }}>
                        Selected Vehicle Record
                      </span>
                      <span style={{
                        padding: '2px 8px',
                        backgroundColor: '#dbeafe',
                        color: '#1e40af',
                        borderRadius: '9999px',
                        fontSize: '11px',
                        fontWeight: 700
                      }}>
                        {selectedVehicle.vehicle_class?.toUpperCase()}
                      </span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '10px', marginBottom: '1rem' }}>
                      <div>
                        <span style={{ fontSize: '11px', color: '#64748b', display: 'block' }}>Global Vehicle ID</span>
                        <strong style={{ fontSize: '16px', color: '#003366', fontFamily: 'var(--font-mono, monospace)' }}>
                          {selectedVehicle.global_vehicle_id}
                        </strong>
                      </div>
                      <div>
                        <span style={{ fontSize: '11px', color: '#64748b', display: 'block' }}>License Plate</span>
                        <strong style={{ fontSize: '15px', color: '#0f172a', fontFamily: 'var(--font-mono, monospace)' }}>
                          {selectedVehicle.plate_text || 'Plate Not Available'}
                        </strong>
                      </div>
                      <div>
                        <span style={{ fontSize: '11px', color: '#64748b', display: 'block' }}>AI Match Confidence</span>
                        <strong style={{ fontSize: '15px', color: '#0f766e' }}>
                          {selectedVehicle.match_confidence
                            ? (selectedVehicle.match_confidence > 1 ? `${selectedVehicle.match_confidence.toFixed(1)}%` : `${(selectedVehicle.match_confidence * 100).toFixed(1)}%`)
                            : 'N/A'}
                        </strong>
                      </div>
                      <div>
                        <span style={{ fontSize: '11px', color: '#64748b', display: 'block' }}>Cameras Detected</span>
                        <strong style={{ fontSize: '15px', color: '#003366' }}>
                          {selectedVehicle.camera_count || selectedVehicle.camera_observations?.length || 1}
                        </strong>
                      </div>
                      <div>
                        <span style={{ fontSize: '11px', color: '#64748b', display: 'block' }}>First Seen</span>
                        <strong style={{ fontSize: '13px', color: '#334155' }}>
                          {selectedVehicle.first_seen_cam || 'N/A'}
                        </strong>
                      </div>
                      <div>
                        <span style={{ fontSize: '11px', color: '#64748b', display: 'block' }}>Last Seen</span>
                        <strong style={{ fontSize: '13px', color: '#334155' }}>
                          {selectedVehicle.last_seen_cam || 'N/A'}
                        </strong>
                      </div>
                    </div>

                    {/* Camera Sequence Badges */}
                    <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem' }}>
                      <span style={{ fontSize: '11px', fontWeight: 700, color: '#475569', display: 'block', marginBottom: '6px' }}>
                        Camera Sequence:
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                        {selectedVehicle.camera_observations && selectedVehicle.camera_observations.length > 0 ? (
                          selectedVehicle.camera_observations.map((camId, idx) => (
                            <React.Fragment key={camId}>
                              <span style={{
                                padding: '4px 10px',
                                backgroundColor: '#003366',
                                color: '#ffffff',
                                borderRadius: '6px',
                                fontSize: '11.5px',
                                fontWeight: 700,
                                fontFamily: 'var(--font-mono, monospace)',
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '4px'
                              }}>
                                <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#38bdf8' }} />
                                {camId}
                              </span>
                              {idx < selectedVehicle.camera_observations.length - 1 && (
                                <span style={{ color: '#0284c7', fontWeight: 800, fontSize: '14px' }}>→</span>
                              )}
                            </React.Fragment>
                          ))
                        ) : (
                          <span style={{ fontSize: '12px', color: '#64748b' }}>No camera observations recorded</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Vehicle Timeline */}
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <h4 style={{ fontSize: '13px', fontWeight: 800, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>
                        Vehicle Observation Timeline
                      </h4>
                      {trajectoryPolyline.length > 1 && (
                        <span style={{ fontSize: '11px', color: '#0284c7', fontWeight: 700 }}>
                          ✓ Highlighted on GIS Map
                        </span>
                      )}
                    </div>

                    <div style={{
                      backgroundColor: '#ffffff',
                      border: '1px solid #e2e8f0',
                      borderRadius: '10px',
                      padding: '1rem',
                      maxHeight: '220px',
                      overflowY: 'auto'
                    }}>
                      {selectedVehicle.timeline && selectedVehicle.timeline.length > 0 ? (
                        <div style={{ position: 'relative', paddingLeft: '1rem' }}>
                          {/* Timeline vertical bar */}
                          <div style={{ position: 'absolute', left: '19px', top: '10px', bottom: '10px', width: '2px', backgroundColor: '#cbd5e1' }} />

                          {selectedVehicle.timeline.map((step, idx) => (
                            <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', marginBottom: '12px', position: 'relative' }}>
                              <div style={{
                                width: '10px',
                                height: '10px',
                                borderRadius: '9999px',
                                backgroundColor: idx === 0 ? '#10b981' : idx === selectedVehicle.timeline.length - 1 ? '#dc2626' : '#0284c7',
                                border: '2px solid #ffffff',
                                boxShadow: '0 0 4px rgba(0,0,0,0.2)',
                                marginTop: '4px',
                                zIndex: 2
                              }} />
                              <div style={{ flex: 1 }}>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                  <strong style={{ fontSize: '12.5px', color: '#003366', fontFamily: 'var(--font-mono, monospace)' }}>
                                    {step.camera_id}
                                  </strong>
                                  <span style={{
                                    fontSize: '10.5px',
                                    fontWeight: 700,
                                    padding: '1px 6px',
                                    borderRadius: '4px',
                                    backgroundColor: idx === 0 ? '#ecfdf5' : idx === selectedVehicle.timeline.length - 1 ? '#fef2f2' : '#f0f9ff',
                                    color: idx === 0 ? '#065f46' : idx === selectedVehicle.timeline.length - 1 ? '#991b1b' : '#0369a1'
                                  }}>
                                    {step.status}
                                  </span>
                                </div>
                                <div style={{ fontSize: '11.5px', color: '#475569' }}>
                                  {step.road_name}
                                </div>
                                <div style={{ fontSize: '11px', color: '#94a3b8', fontFamily: 'var(--font-mono, monospace)' }}>
                                  Timestamp: {step.timestamp}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ padding: '1rem', textAlign: 'center', color: '#64748b', fontSize: '12px' }}>
                          No chronological timeline sightings available for this vehicle.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ padding: '3rem', textAlign: 'center', color: '#64748b', border: '1px dashed #cbd5e1', borderRadius: '12px' }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '36px', color: '#94a3b8', marginBottom: '8px' }}>search</span>
                  <p style={{ margin: 0, fontSize: '13px' }}>Select a vehicle from the table or search above to view detailed trajectory and timeline intelligence.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* 4. TRAFFIC ALERTS & INCIDENT QUEUE SECTION (Preserved) */}
      <section className="container-7xl" style={{ paddingBottom: '2.5rem' }}>
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          border: '1px solid #e2e8f0',
          padding: '1.5rem',
          boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 800, color: '#003366', margin: 0 }}>
                Corridor Enforcement Alerts &amp; Incidents
              </h3>
              <p style={{ fontSize: '12px', color: '#64748b', margin: '2px 0 0 0' }}>
                Active safety alerts, speed violations, and flow restrictions across surveillance sectors.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setRoute && setRoute('/tracking')}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                backgroundColor: '#ffffff',
                color: '#003366',
                fontWeight: 700,
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              View Trajectory Map →
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
            {alerts.slice(0, 4).map((alert) => (
              <div
                key={alert.id}
                style={{
                  backgroundColor: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  padding: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between'
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <strong style={{ fontSize: '12.5px', color: '#003366' }}>{alert.type}</strong>
                    <span style={{
                      fontSize: '10px',
                      fontWeight: 700,
                      padding: '1px 6px',
                      borderRadius: '4px',
                      backgroundColor: alert.severity === 'Critical' ? '#fee2e2' : '#fef3c7',
                      color: alert.severity === 'Critical' ? '#991b1b' : '#92400e',
                      textTransform: 'uppercase'
                    }}>
                      {alert.severity}
                    </span>
                  </div>
                  <div style={{ fontSize: '12px', color: '#334155', fontFamily: 'var(--font-mono, monospace)', fontWeight: 700 }}>
                    {alert.plate}
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>
                    {alert.location}
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '8px', fontSize: '10.5px', color: '#94a3b8', borderTop: '1px solid #f1f5f9', paddingTop: '6px' }}>
                  <span>{alert.time}</span>
                  <span style={{ fontWeight: 600, color: '#475569' }}>{alert.status}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
