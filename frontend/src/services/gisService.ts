/**
 * NETRA GIS Service
 * Networked Engine for Traffic Recognition & Analytics
 * 
 * Provides GIS camera telemetry, multi-camera vehicle trajectory tracking,
 * and traffic status layers. Structured for seamless migration to FastAPI REST & WebSockets.
 */

export interface Camera {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  status: 'online' | 'warning' | 'offline';
  lastDetection?: string;
  vehicleCount?: number;
  locationName?: string;
  roadName?: string;
  road_name?: string;
  speedLimitKmh?: number;
  speed_limit_kmh?: number;
  speedAvg?: string;
  fps?: number;
  city?: string;
  state?: string;
}

export interface VehicleTimelineItem {
  camera_id: string;
  camera_name?: string;
  road_name?: string;
  timestamp: string;
  status: string;
  confidence?: number;
}

export interface GlobalVehicleItem {
  global_vehicle_id: string;
  vehicle_class: string;
  plate_text?: string | null;
  match_confidence?: number;
  camera_observations: string[];
  first_seen_cam?: string | null;
  last_seen_cam?: string | null;
  camera_count?: number;
  timeline?: VehicleTimelineItem[];
  created_at?: string;
}


export interface VehicleDetection {
  cameraId: string;
  cameraName?: string;
  latitude: number;
  longitude: number;
  timestamp: string;
  plate: string;
  vehicleType: string;
  confidence: number;
  direction?: string;
  speed?: string;
}

export interface VehicleTrajectory {
  plate: string;
  detections: VehicleDetection[];
  totalDistanceKm?: number;
  avgSpeedKmH?: number;
  status?: string;
}

export interface TrafficZone {
  id: string;
  name: string;
  status: 'normal' | 'congestion' | 'heavy' | 'incident';
  statusText: string;
  color: string;
  center: [number, number];
  radiusMeters: number;
  density: 'Low' | 'Moderate' | 'High' | 'Restricted' | 'Optimal';
  avgSpeed: string;
}

export interface TrafficIncident {
  id: string;
  title: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  latitude: number;
  longitude: number;
  timestamp: string;
  corridor: string;
  description: string;
}

/**
 * Normalizes vehicle registration strings (strips spaces and hyphens, uppercase)
 */
export function normalizePlate(plate: string): string {
  return (plate || '').replace(/[\s\-_]/g, '').toUpperCase();
}


export interface PlateCapture {
  id: number;
  plate: string;
  type: string;
  camId: string;
  timestamp: string;
  confidence: string;
  status: string;
}

export interface AlertItem {
  id: string;
  type: string;
  plate: string;
  location: string;
  severity: string;
  time: string;
  status: string;
}

// ============================================================================
// SERVICE API METHODS (Pluggable for FastAPI backend)
// ============================================================================

const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || '';

/**
 * Register a new camera node in PostgreSQL/PostGIS through FastAPI
 */
export async function createCamera(cameraData: {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  status?: string;
  road_name?: string;
  speed_limit_kmh?: number;
  city?: string;
  state?: string;
  location_name?: string;
}): Promise<Camera> {
  const payload = {
    id: cameraData.id.trim(),
    name: cameraData.name.trim(),
    latitude: cameraData.latitude,
    longitude: cameraData.longitude,
    status: cameraData.status || 'online',
    road_name: cameraData.road_name?.trim() || null,
    speed_limit_kmh: cameraData.speed_limit_kmh ? Number(cameraData.speed_limit_kmh) : null,
    city: cameraData.city?.trim() || 'Coimbatore',
    state: cameraData.state?.trim() || 'Tamil Nadu',
    location_name: cameraData.location_name?.trim() || `${cameraData.road_name || cameraData.name}, ${cameraData.city || 'Coimbatore'}, ${cameraData.state || 'Tamil Nadu'}`
  };

  const res = await fetch(`${API_BASE_URL}/api/v1/cameras`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    let errDetail = 'Failed to register camera node';
    try {
      const errJson = await res.json();
      if (errJson?.detail) errDetail = errJson.detail;
    } catch (_) {
      errDetail = await res.text();
    }
    throw new Error(errDetail);
  }

  return await res.json();
}

/**
 * Fetch global canonical vehicle records from backend
 */
export async function getGlobalVehicles(limit: number = 50, search?: string): Promise<GlobalVehicleItem[]> {
  try {
    const params = new URLSearchParams();
    params.set('limit', limit.toString());
    if (search && search.trim()) {
      params.set('search', search.trim());
    }
    const res = await fetch(`${API_BASE_URL}/api/v1/vehicles?${params.toString()}`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) return data;
    }
  } catch (e) {
    console.warn('API error fetching global vehicles', e);
  }
  return [];
}

/**
 * Fetch list of all operational cameras in the grid
 */
export async function getCameras(): Promise<Camera[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/cameras`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) return data;
    }
  } catch (e) {
    try {
      const resCompat = await fetch(`${API_BASE_URL}/api/cameras`);
      if (resCompat.ok) {
        const data = await resCompat.json();
        if (Array.isArray(data)) return data;
      }
    } catch (err) {
      console.error('API error fetching cameras from backend:', err);
    }
  }
  return [];
}

/**
 * Fetch vehicle detection trajectory for a given license plate or global vehicle ID from FastAPI backend
 */
export async function getVehicleTrajectory(vehicleNumber: string): Promise<VehicleTrajectory | null> {
  if (!vehicleNumber || !vehicleNumber.trim()) return null;

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/vehicles/${encodeURIComponent(vehicleNumber.trim())}/trajectory`);
    if (res.ok) {
      const data = await res.json();
      if (data && Array.isArray(data.detections) && data.detections.length > 0) {
        return data;
      }
    }
  } catch (_) {
    try {
      const resCompat = await fetch(`${API_BASE_URL}/api/vehicles/${encodeURIComponent(vehicleNumber.trim())}/trajectory`);
      if (resCompat.ok) {
        const data = await resCompat.json();
        if (data && Array.isArray(data.detections) && data.detections.length > 0) {
          return data;
        }
      }
    } catch (e) {
      console.warn('API error fetching vehicle trajectory', e);
    }
  }

  // Never fabricate synthetic coordinates or fake trajectories
  return null;
}

/**
 * Search global vehicles matching a vehicle ID or plate query from FastAPI backend
 */
export async function findGlobalVehicle(query: string): Promise<GlobalVehicleItem | null> {
  if (!query || !query.trim()) return null;
  const clean = query.trim().toUpperCase();
  const norm = normalizePlate(clean);

  const results = await getGlobalVehicles(15, query.trim());
  if (!results || results.length === 0) return null;

  // Exact ID match
  const exactId = results.find(v => (v.global_vehicle_id || '').toUpperCase() === clean);
  if (exactId) return exactId;

  // Exact plate match
  const exactPlate = results.find(v => normalizePlate(v.plate_text || '') === norm);
  if (exactPlate) return exactPlate;

  // Partial match
  const partial = results.find(v => {
    const p = normalizePlate(v.plate_text || '');
    return p && (p.includes(norm) || norm.includes(p));
  });
  if (partial) return partial;

  return results[0] || null;
}

/**
 * Fetch active traffic zones, density corridors, and incidents
 */
export async function getTrafficEvents(): Promise<{
  zones: TrafficZone[];
  incidents: TrafficIncident[];
}> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/traffic/events`);
    if (res.ok) {
      const data = await res.json();
      if (data && (Array.isArray(data.zones) || Array.isArray(data.incidents))) {
        return {
          zones: Array.isArray(data.zones) ? data.zones : [],
          incidents: Array.isArray(data.incidents) ? data.incidents : []
        };
      }
    }
  } catch (e) {
    try {
      const resCompat = await fetch(`${API_BASE_URL}/api/v1/traffic/events`);
      if (resCompat.ok) {
        const data = await resCompat.json();
        if (data && (Array.isArray(data.zones) || Array.isArray(data.incidents))) {
          return {
            zones: Array.isArray(data.zones) ? data.zones : [],
            incidents: Array.isArray(data.incidents) ? data.incidents : []
          };
        }
      }
    } catch (err) {
      console.error('API error fetching traffic events from backend:', err);
    }
  }

  return {
    zones: [],
    incidents: []
  };
}

/**
 * Fetch recent ANPR plate sightings from backend
 */
export async function getPlateCaptures(limit: number = 20): Promise<PlateCapture[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/plates/captures?limit=${limit}`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) return data;
    }
  } catch (e) {
    try {
      const resCompat = await fetch(`${API_BASE_URL}/api/plates/captures?limit=${limit}`);
      if (resCompat.ok) {
        const data = await resCompat.json();
        if (Array.isArray(data)) return data;
      }
    } catch (err) {
      console.error('API error fetching plate captures from backend:', err);
    }
  }
  return [];
}

/**
 * Fetch active enforcement alerts from backend
 */
export async function getAlerts(): Promise<AlertItem[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/alerts`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) return data;
    }
  } catch (e) {
    try {
      const resCompat = await fetch(`${API_BASE_URL}/api/alerts`);
      if (resCompat.ok) {
        const data = await resCompat.json();
        if (Array.isArray(data)) return data;
      }
    } catch (err) {
      console.error('API error fetching alerts from backend:', err);
    }
  }
  return [];
}

/**
 * Fetch real traffic analytics summary from backend
 */
export async function getTrafficAnalytics(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/analytics/summary`);
    if (res.ok) {
      const data = await res.json();
      if (data && (data.vehicle_class_statistics || data.total_global_vehicles || data.route_statistics)) {
        return data;
      }
    }
  } catch (e) {
    console.warn('API error fetching traffic analytics', e);
  }
  return null;
}

/**
 * Fetch OD matrix data from backend
 */
export async function getOdMatrix(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/analytics/od-matrix`);
    if (res.ok) {
      return await res.json();
    }
  } catch (e) {
    console.warn('API error fetching OD matrix', e);
  }
  return null;
}

/**
 * Architecture hook for future WebSocket real-time updates from FastAPI / AI Engine
 */
export function subscribeToRealTimeUpdates(
  channel: 'cameras' | 'trajectories' | 'incidents',
  onMessage: (data: any) => void
): () => void {
  // Real-time updates subscription hook (WebSocket integration point)
  return () => {};
}
