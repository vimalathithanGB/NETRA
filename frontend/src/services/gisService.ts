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

// ============================================================================
// MOCK GIS DATASETS (Coimbatore Urban Traffic Network — Tamil Nadu)
// ============================================================================

export const MOCK_CAMERAS: Camera[] = [
  {
    id: 'CAM_01',
    name: 'CAM_01 - SIMULATED_ROAD_A',
    latitude: 11.0168,
    longitude: 76.9558,
    status: 'online',
    lastDetection: '14:32:10',
    vehicleCount: 31,
    locationName: 'Ingress Corridor Node (Avinashi Road)',
    speedAvg: '38 km/h',
    fps: 30
  },
  {
    id: 'CAM_02',
    name: 'CAM_02 - SIMULATED_ROAD_B',
    latitude: 11.0192,
    longitude: 76.9581,
    status: 'online',
    lastDetection: '14:33:04',
    vehicleCount: 24,
    locationName: 'Mid-Corridor Surveillance Node (Gandhipuram)',
    speedAvg: '42 km/h',
    fps: 30
  },
  {
    id: 'CAM_03',
    name: 'CAM_03 - SIMULATED_ROAD_C',
    latitude: 11.0225,
    longitude: 76.9610,
    status: 'online',
    lastDetection: '14:30:15',
    vehicleCount: 17,
    locationName: 'Egress Sensor Node (Sathy Road)',
    speedAvg: '40 km/h',
    fps: 30
  }
];

export const MOCK_TRAFFIC_ZONES: TrafficZone[] = [
  {
    id: 'north-zone',
    name: 'Sathy Road North (CAM_03)',
    status: 'normal',
    statusText: 'Normal Flow (40 km/h avg)',
    color: '#138808',
    center: [11.0225, 76.9610],
    radiusMeters: 500,
    density: 'Low',
    avgSpeed: '40 km/h'
  },
  {
    id: 'central-hub',
    name: 'Gandhipuram Central Hub (CAM_02)',
    status: 'congestion',
    statusText: 'Congestion Warning (18 km/h)',
    color: '#f59e0b',
    center: [11.0192, 76.9581],
    radiusMeters: 600,
    density: 'High',
    avgSpeed: '18 km/h'
  },
  {
    id: 'east-corridor',
    name: 'Avinashi Road Ingress (CAM_01)',
    status: 'normal',
    statusText: 'Smooth (42 km/h avg)',
    color: '#138808',
    center: [11.0168, 76.9558],
    radiusMeters: 550,
    density: 'Optimal',
    avgSpeed: '42 km/h'
  },
  {
    id: 'port-access',
    name: 'Trichy Road Junction',
    status: 'incident',
    statusText: 'Incident Reported (Lane 2 Blocked)',
    color: '#ba1a1a',
    center: [11.0012, 76.9680],
    radiusMeters: 600,
    density: 'Restricted',
    avgSpeed: '10 km/h'
  }
];

export const MOCK_INCIDENTS: TrafficIncident[] = [
  {
    id: 'INC-7802',
    title: 'Lane 2 Bottleneck & Commercial Carrier Breakdown',
    severity: 'high',
    latitude: 11.0168,
    longitude: 76.9558,
    timestamp: '14:05 IST',
    corridor: 'Avinashi Road Ingress Corridor',
    description: 'Commercial vehicle disabled on outbound arterial near CAM_01. Traffic patrol dispatched.'
  },
  {
    id: 'INC-7805',
    title: 'High Density Crawl at Flyover Entry',
    severity: 'medium',
    latitude: 11.0192,
    longitude: 76.9581,
    timestamp: '14:28 IST',
    corridor: 'Gandhipuram Flyover Junction (CAM_02)',
    description: 'Vehicle density spike above 85% threshold on central arterial.'
  }
];

// Mock trajectories mapped by normalized plate number (CAM_01 -> CAM_02 -> CAM_03 corridor)
export const MOCK_TRAJECTORIES: Record<string, VehicleTrajectory> = {
  'GV_000001': {
    plate: 'GV_000001',
    status: 'Active Track',
    totalDistanceKm: 0.9,
    avgSpeedKmH: 42.0,
    detections: [
      {
        cameraId: 'CAM_01',
        cameraName: 'CAM_01 - SIMULATED_ROAD_A',
        latitude: 11.0168,
        longitude: 76.9558,
        timestamp: '14:32:10',
        plate: 'GV_000001',
        vehicleType: 'bus',
        confidence: 99.4,
        direction: 'North-East',
        speed: '40 km/h'
      },
      {
        cameraId: 'CAM_02',
        cameraName: 'CAM_02 - SIMULATED_ROAD_B',
        latitude: 11.0192,
        longitude: 76.9581,
        timestamp: '14:34:25',
        plate: 'GV_000001',
        vehicleType: 'bus',
        confidence: 98.9,
        direction: 'North-East',
        speed: '42 km/h'
      },
      {
        cameraId: 'CAM_03',
        cameraName: 'CAM_03 - SIMULATED_ROAD_C',
        latitude: 11.0225,
        longitude: 76.9610,
        timestamp: '14:36:40',
        plate: 'GV_000001',
        vehicleType: 'bus',
        confidence: 99.2,
        direction: 'North-East',
        speed: '40 km/h'
      }
    ]
  },
  'TN38AB1234': {
    plate: 'TN38AB1234',
    status: 'Active Track',
    totalDistanceKm: 0.9,
    avgSpeedKmH: 44.5,
    detections: [
      {
        cameraId: 'CAM_01',
        cameraName: 'CAM_01 - Avinashi Road Ingress',
        latitude: 11.0168,
        longitude: 76.9558,
        timestamp: '14:32:10',
        plate: 'TN38AB1234',
        vehicleType: 'Car',
        confidence: 99.4,
        direction: 'North-East',
        speed: '42 km/h'
      },
      {
        cameraId: 'CAM_02',
        cameraName: 'CAM_02 - Gandhipuram Mid-Corridor',
        latitude: 11.0192,
        longitude: 76.9581,
        timestamp: '14:34:18',
        plate: 'TN38AB1234',
        vehicleType: 'Car',
        confidence: 98.9,
        direction: 'North-East',
        speed: '46 km/h'
      },
      {
        cameraId: 'CAM_03',
        cameraName: 'CAM_03 - Sathy Road Egress',
        latitude: 11.0225,
        longitude: 76.9610,
        timestamp: '14:36:02',
        plate: 'TN38AB1234',
        vehicleType: 'Car',
        confidence: 99.1,
        direction: 'North-East',
        speed: '44 km/h'
      }
    ]
  },
  'ONDUTY': {
    plate: 'ONDUTY',
    status: 'Active Track',
    totalDistanceKm: 0.9,
    avgSpeedKmH: 48.0,
    detections: [
      {
        cameraId: 'CAM_01',
        cameraName: 'CAM_01 - Avinashi Road Ingress',
        latitude: 11.0168,
        longitude: 76.9558,
        timestamp: '14:30:00',
        plate: 'ONDUTY',
        vehicleType: 'Patrol Vehicle',
        confidence: 99.8,
        direction: 'North-East',
        speed: '45 km/h'
      },
      {
        cameraId: 'CAM_02',
        cameraName: 'CAM_02 - Gandhipuram Mid-Corridor',
        latitude: 11.0192,
        longitude: 76.9581,
        timestamp: '14:32:05',
        plate: 'ONDUTY',
        vehicleType: 'Patrol Vehicle',
        confidence: 99.6,
        direction: 'North-East',
        speed: '50 km/h'
      },
      {
        cameraId: 'CAM_03',
        cameraName: 'CAM_03 - Sathy Road Egress',
        latitude: 11.0225,
        longitude: 76.9610,
        timestamp: '14:33:50',
        plate: 'ONDUTY',
        vehicleType: 'Patrol Vehicle',
        confidence: 99.7,
        direction: 'North-East',
        speed: '48 km/h'
      }
    ]
  },
  'DL04C8891': {
    plate: 'DL 04 C 8891',
    status: 'Active Track',
    totalDistanceKm: 0.9,
    avgSpeedKmH: 46.2,
    detections: [
      {
        cameraId: 'CAM_01',
        cameraName: 'CAM_01 - Ingress Node',
        latitude: 11.0168,
        longitude: 76.9558,
        timestamp: '14:15:30',
        plate: 'DL 04 C 8891',
        vehicleType: 'Commercial SUV',
        confidence: 99.5,
        direction: 'North-East',
        speed: '44 km/h'
      },
      {
        cameraId: 'CAM_02',
        cameraName: 'CAM_02 - Mid-Corridor Node',
        latitude: 11.0192,
        longitude: 76.9581,
        timestamp: '14:18:14',
        plate: 'DL 04 C 8891',
        vehicleType: 'Commercial SUV',
        confidence: 99.2,
        direction: 'North-East',
        speed: '46 km/h'
      },
      {
        cameraId: 'CAM_03',
        cameraName: 'CAM_03 - Egress Node',
        latitude: 11.0225,
        longitude: 76.9610,
        timestamp: '14:21:00',
        plate: 'DL 04 C 8891',
        vehicleType: 'Commercial SUV',
        confidence: 99.8,
        direction: 'North-East',
        speed: '48 km/h'
      }
    ]
  },
  'HR26DK5092': {
    plate: 'HR 26 DK 5092',
    status: 'Active Track',
    totalDistanceKm: 0.9,
    avgSpeedKmH: 45.0,
    detections: [
      {
        cameraId: 'CAM_01',
        cameraName: 'CAM_01 - Ingress Node',
        latitude: 11.0168,
        longitude: 76.9558,
        timestamp: '14:12:00',
        plate: 'HR 26 DK 5092',
        vehicleType: 'Sedan',
        confidence: 98.9,
        direction: 'North-East',
        speed: '45 km/h'
      },
      {
        cameraId: 'CAM_02',
        cameraName: 'CAM_02 - Mid-Corridor Node',
        latitude: 11.0192,
        longitude: 76.9581,
        timestamp: '14:14:22',
        plate: 'HR 26 DK 5092',
        vehicleType: 'Sedan',
        confidence: 99.1,
        direction: 'North-East',
        speed: '46 km/h'
      },
      {
        cameraId: 'CAM_03',
        cameraName: 'CAM_03 - Egress Node',
        latitude: 11.0225,
        longitude: 76.9610,
        timestamp: '14:16:40',
        plate: 'HR 26 DK 5092',
        vehicleType: 'Sedan',
        confidence: 98.9,
        direction: 'North-East',
        speed: '44 km/h'
      }
    ]
  }
};

/**
 * Normalizes vehicle registration strings (strips spaces and hyphens, uppercase)
 */
export function normalizePlate(plate: string): string {
  return (plate || '').replace(/[\s\-_]/g, '').toUpperCase();
}

/**
 * Dynamically generates a realistic trajectory for any arbitrary vehicle number
 * entered by the operator if not present in the preset mock table.
 */
function generateDynamicTrajectory(rawPlate: string): VehicleTrajectory {
  const displayPlate = rawPlate.toUpperCase().trim() || 'GV_000001';

  const baseCameras = [
    MOCK_CAMERAS[0], // CAM_01
    MOCK_CAMERAS[1], // CAM_02
    MOCK_CAMERAS[2]  // CAM_03
  ];

  const now = new Date();
  const detections: VehicleDetection[] = baseCameras.map((cam, idx) => {
    const minutesAgo = (baseCameras.length - 1 - idx) * 3;
    const time = new Date(now.getTime() - minutesAgo * 60 * 1000);
    const timeStr = time.toTimeString().split(' ')[0];

    return {
      cameraId: cam.id,
      cameraName: cam.name,
      latitude: cam.latitude,
      longitude: cam.longitude,
      timestamp: timeStr,
      plate: displayPlate,
      vehicleType: 'Passenger Vehicle',
      confidence: +(97.5 + (idx % 3) * 0.8).toFixed(1),
      direction: 'North-East',
      speed: `${40 + idx * 2} km/h`
    };
  });

  return {
    plate: displayPlate,
    status: 'Active Track',
    totalDistanceKm: 0.9,
    avgSpeedKmH: 42.0,
    detections
  };
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
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch (e) {
    try {
      const resCompat = await fetch(`${API_BASE_URL}/api/cameras`);
      if (resCompat.ok) {
        const data = await resCompat.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      console.warn('API error, falling back to mock cameras', err);
    }
  }
  return [...MOCK_CAMERAS];
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
      if (data && (data.zones?.length || data.incidents?.length)) return data;
    }
  } catch (e) {
    console.warn('API error, falling back to mock traffic events', e);
  }

  return {
    zones: [...MOCK_TRAFFIC_ZONES],
    incidents: [...MOCK_INCIDENTS]
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
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch (e) {
    console.warn('API error, falling back to mock plate captures', e);
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
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch (e) {
    console.warn('API error, falling back to mock alerts', e);
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
  // Currently simulated heartbeat; can be replaced with new WebSocket(...) in future
  const interval = setInterval(() => {
    // Simulated live tick
  }, 10000);

  return () => clearInterval(interval);
}

