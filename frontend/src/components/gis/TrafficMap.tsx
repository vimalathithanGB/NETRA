import React, { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet's default icon URL issue in bundlers
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png'
});

export interface TrafficMapProps {
  center?: [number, number];
  zoom?: number;
  minZoom?: number;
  maxZoom?: number;
  height?: string | number;
  children?: React.ReactNode;
  onMapClick?: (coords: { lat: number; lng: number }) => void;
  showResetButton?: boolean;
  tileLayerType?: 'standard' | 'satellite' | 'dark';
  autoFitCoords?: [number, number][];
  className?: string;
  style?: React.CSSProperties;
}

// Controller component to listen for click events and resize maps
function MapEventsController({
  onMapClick,
  autoFitCoords,
  defaultCenter,
  defaultZoom,
  resetTrigger
}: {
  onMapClick?: (coords: { lat: number; lng: number }) => void;
  autoFitCoords?: [number, number][];
  defaultCenter: [number, number];
  defaultZoom: number;
  resetTrigger: number;
}) {
  const map = useMap();

  useMapEvents({
    click(e) {
      if (onMapClick) {
        onMapClick({ lat: e.latlng.lat, lng: e.latlng.lng });
      }
    }
  });

  // Handle auto-fit when coordinates are provided (e.g. For vehicle trajectory)
  useEffect(() => {
    if (autoFitCoords && autoFitCoords.length > 0) {
      try {
        const bounds = L.latLngBounds(autoFitCoords);
        if (bounds.isValid()) {
          map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
        }
      } catch (err) {
        console.warn('Error fitting bounds:', err);
      }
    }
  }, [map, autoFitCoords]);

  // Handle reset button trigger
  useEffect(() => {
    if (resetTrigger > 0) {
      if (autoFitCoords && autoFitCoords.length > 0) {
        try {
          const bounds = L.latLngBounds(autoFitCoords);
          if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
            return;
          }
        } catch (err) {
          console.warn('Error fitting bounds on reset:', err);
        }
      }
      map.flyTo(defaultCenter, defaultZoom, { duration: 0.8 });
    }
  }, [resetTrigger, map, defaultCenter, defaultZoom, autoFitCoords]);

  // Invalidate size on mount to prevent gray tiles
  useEffect(() => {
    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 250);
    return () => clearTimeout(timer);
  }, [map]);

  return null;
}

export const TrafficMap: React.FC<TrafficMapProps> = ({
  center = [11.0183, 76.9580] as [number, number], // Default: Coimbatore Urban Center, Tamil Nadu
  zoom = 14,
  minZoom = 5,
  maxZoom = 18,
  height = '100%',
  children,
  onMapClick,
  showResetButton = true,
  tileLayerType = 'standard',
  autoFitCoords,
  className = '',
  style = {}
}) => {
  const [resetCount, setResetCount] = useState(0);
  const [lastClickedPoint, setLastClickedPoint] = useState<{ lat: number; lng: number } | null>(null);

  // Available tile providers
  const tileLayers: Record<'standard' | 'satellite' | 'dark', { url: string; attribution: string }> = {
    standard: {
      url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    },
    satellite: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attribution: '&copy; Esri, Maxar, Earthstar Geographics'
    },
    dark: {
      url: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
      attribution: '&copy; CARTO &copy; OpenStreetMap'
    }
  };

  const currentTiles = tileLayers[tileLayerType] || tileLayers.standard;

  const handleMapClick = (coords: { lat: number; lng: number }) => {
    setLastClickedPoint(coords);
    if (onMapClick) {
      onMapClick(coords);
    }
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: height,
        borderRadius: 'inherit',
        overflow: 'hidden',
        ...style
      }}
      className={className}
    >
      <MapContainer
        center={center}
        zoom={zoom}
        minZoom={minZoom}
        maxZoom={maxZoom}
        style={{ width: '100%', height: '100%' }}
        zoomControl={true}
      >
        <TileLayer
          url={currentTiles.url}
          attribution={currentTiles.attribution}
        />

        <MapEventsController
          onMapClick={handleMapClick}
          autoFitCoords={autoFitCoords}
          defaultCenter={center}
          defaultZoom={zoom}
          resetTrigger={resetCount}
        />

        {children}
      </MapContainer>

      {/* Floating Map Controls & Reset View Button */}
      {showResetButton && (
        <div
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <button
            type="button"
            onClick={() => setResetCount((c) => c + 1)}
            title="Reset Map to Default Sector View"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              backgroundColor: 'rgba(255, 255, 255, 0.94)',
              backdropFilter: 'blur(6px)',
              border: '1px solid #cbd5e1',
              color: '#003366',
              borderRadius: '6px',
              padding: '6px 10px',
              fontSize: '11.5px',
              fontWeight: 700,
              boxShadow: '0 2px 8px rgba(0, 30, 60, 0.12)',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#ffffff';
              e.currentTarget.style.transform = 'translateY(-1px)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.94)';
              e.currentTarget.style.transform = 'none';
            }}
          >
            <svg style={{ width: '14px', height: '14px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            <span>Reset Map</span>
          </button>
        </div>
      )}

      {/* Clicked Coordinate Toast Pill (Unobtrusive) */}
      {lastClickedPoint && (
        <div
          style={{
            position: 'absolute',
            bottom: '12px',
            right: '12px',
            zIndex: 999,
            backgroundColor: 'rgba(0, 30, 64, 0.88)',
            backdropFilter: 'blur(6px)',
            color: '#ffffff',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '10.5px',
            fontFamily: 'var(--font-mono, monospace)',
            pointerEvents: 'none',
            border: '1px solid rgba(255, 255, 255, 0.15)'
          }}
        >
          GPS: {lastClickedPoint.lat.toFixed(4)}°N, {lastClickedPoint.lng.toFixed(4)}°E
        </div>
      )}
    </div>
  );
};

export default TrafficMap;
