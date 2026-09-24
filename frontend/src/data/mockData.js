// Mock datasets representing real-time telemetry from NETRA intelligent grid

export const initialMetrics = {
  activeCameras: 1428,
  totalCameras: 1452,
  cameraOnlinePercent: 98.4,
  camerasAddedToday: 12,
  vehiclesDetected: "4.82M",
  peakTime: "09:30 IST",
  vehicleGrowthPercent: "+5.4%",
  anprAccuracy: "99.2%",
  anprModel: "Model v4.8-Gov",
  activeIncidents: 14,
  incidentsPendingReview: 14
};

export const initialZones = [
  {
    id: 'north-zone',
    name: 'Sathy Road North (CAM_03)',
    status: 'optimal',
    statusText: 'Optimal (40 km/h avg)',
    color: '#138808',
    lat: '11.0225',
    lng: '76.9610',
    density: 'Low'
  },
  {
    id: 'central-hub',
    name: 'Gandhipuram Central (CAM_02)',
    status: 'warning',
    statusText: 'Congestion Warning (18 km/h)',
    color: '#f59e0b',
    lat: '11.0192',
    lng: '76.9581',
    density: 'High'
  },
  {
    id: 'east-corridor',
    name: 'Avinashi Road Ingress (CAM_01)',
    status: 'smooth',
    statusText: 'Smooth (42 km/h avg)',
    color: '#138808',
    lat: '11.0168',
    lng: '76.9558',
    density: 'Optimal'
  },
  {
    id: 'port-access',
    name: 'Trichy Road Junction',
    status: 'incident',
    statusText: 'Incident Reported (Lane 2 Blocked)',
    color: '#ba1a1a',
    lat: '11.0012',
    lng: '76.9680',
    density: 'Restricted'
  }
];

export const initialAnprCaptures = [
  {
    id: 1,
    plate: 'DL 04 C 8891',
    type: 'Commercial SUV',
    camId: 'CAM-DL-412',
    timestamp: '14:22:08',
    confidence: '99.8%',
    status: 'Verified'
  },
  {
    id: 2,
    plate: 'HR 26 DK 5092',
    type: 'Sedan',
    camId: 'CAM-HR-109',
    timestamp: '14:22:07',
    confidence: '98.9%',
    status: 'Verified'
  },
  {
    id: 3,
    plate: 'UP 16 F 3301',
    type: 'Heavy Truck',
    camId: 'CAM-UP-088',
    timestamp: '14:22:05',
    confidence: '99.4%',
    status: 'Verified'
  },
  {
    id: 4,
    plate: 'KA 01 M 9912',
    type: 'Electric Bus',
    camId: 'CAM-DL-301',
    timestamp: '14:22:02',
    confidence: '99.1%',
    status: 'Verified'
  },
  {
    id: 5,
    plate: 'BR 01 AB 1234',
    type: 'Two Wheeler',
    camId: 'CAM-DL-115',
    timestamp: '14:21:59',
    confidence: '94.2%',
    status: 'Review'
  }
];

export const streamingPool = [
  { plate: 'MH 12 QX 4402', type: 'Private Sedan', camId: 'CAM-DL-204', confidence: '99.5%' },
  { plate: 'DL 1C AA 9021', type: 'EV Taxi', camId: 'CAM-DL-119', confidence: '99.7%' },
  { plate: 'RJ 14 TB 3004', type: 'Light Commercial', camId: 'CAM-HR-405', confidence: '97.8%' },
  { plate: 'UP 14 CZ 7811', type: 'Two Wheeler', camId: 'CAM-UP-112', confidence: '96.2%' },
  { plate: 'CH 01 BG 6599', type: 'SUV', camId: 'CAM-DL-802', confidence: '99.2%' },
  { plate: 'WB 02 K 5519', type: 'Freight Carrier', camId: 'CAM-DL-310', confidence: '98.6%' }
];

export const systemHealth = {
  serverLoad: 34.2,
  serverLoadCluster: 'Cluster A (Coimbatore Node)',
  bandwidth: '1.84 TB/s',
  bandwidthPct: 62,
  encryptionStandard: '256-Bit SSL',
  encryptionStatus: 'Zero Vulnerabilities (ISO/IEC 27001)',
  dbSyncStatus: 'Synced (0ms lag)',
  dbLocation: 'State Data Center, Tamil Nadu'
};

export const analyticsData = {
  hourlyVolume: [
    { hour: '00:00', count: 42000 },
    { hour: '03:00', count: 18000 },
    { hour: '06:00', count: 85000 },
    { hour: '09:00', count: 320000 },
    { hour: '12:00', count: 210000 },
    { hour: '15:00', count: 260000 },
    { hour: '18:00', count: 395000 },
    { hour: '21:00', count: 175000 }
  ],
  vehicleClasses: [
    { name: 'Sedan / Hatchback', percentage: 41, color: '#003366' },
    { name: 'Two Wheeler', percentage: 28, color: '#00677d' },
    { name: 'SUV & MUV', percentage: 16, color: '#138808' },
    { name: 'Commercial & Freight', percentage: 11, color: '#ff7a1a' },
    { name: 'Public Transit Bus', percentage: 4, color: '#8b5cf6' }
  ],
  violationsToday: [
    { type: 'Speed Limit Infraction (>80 km/h)', count: 412, trend: '-8%' },
    { type: 'Red Light Jump Violation', count: 289, trend: '-14%' },
    { type: 'Non-Standard Number Plate', count: 94, trend: '+2%' },
    { type: 'Wrong Way Driving', count: 38, trend: '-3%' }
  ]
};

export const cameraNodes = [
  { id: 'CAM_01', location: 'Avinashi Road Ingress Node (CAM_01)', fps: 30, status: 'Active', latency: '4ms', edgeModel: 'NETRA-YOLOv8-Edge' },
  { id: 'CAM_02', location: 'Gandhipuram Mid-Corridor Node (CAM_02)', fps: 30, status: 'Active', latency: '5ms', edgeModel: 'NETRA-YOLOv8-Edge' },
  { id: 'CAM_03', location: 'Sathy Road Egress Node (CAM_03)', fps: 30, status: 'Active', latency: '4ms', edgeModel: 'NETRA-YOLOv8-Edge' },
  { id: 'CAM-TN-104', location: 'Trichy Road Junction Node', fps: 30, status: 'Active', latency: '6ms', edgeModel: 'NETRA-YOLOv8-Edge' },
  { id: 'CAM-TN-105', location: 'Mettupalayam Road Gateway', fps: 30, status: 'Active', latency: '5ms', edgeModel: 'NETRA-YOLOv8-Edge' }
];

export const complianceReports = [
  { id: 'SIH-2026-CR-001', title: 'National Highway Corridor Congestion Index Audit', date: '2026-09-18', officer: 'Superintendent R. Sharma', status: 'Approved' },
  { id: 'SIH-2026-CR-002', title: 'Smart Mobility AI Edge ANPR Calibration Report', date: '2026-09-17', officer: 'Tech Dir. V. Ramanujan', status: 'Verified' },
  { id: 'SIH-2026-CR-003', title: 'Weekly Emergency Corridor Signal Override Log', date: '2026-09-15', officer: 'Traffic Insp. K. Verma', status: 'Archived' }
];
