import React from 'react';
import { Polyline } from 'react-leaflet';
import { VehicleDetection } from '../../services/gisService';

interface TrajectoryLineProps {
  detections: VehicleDetection[];
  color?: string;
  weight?: number;
  dashArray?: string;
}

export const TrajectoryLine: React.FC<TrajectoryLineProps> = ({
  detections,
  color = '#ff7a1a',
  weight = 4,
  dashArray = '8, 6'
}) => {
  if (!detections || detections.length < 2) {
    return null;
  }

  const positions: [number, number][] = detections.map((d) => [d.latitude, d.longitude]);

  return (
    <>
      {/* Background glow / outline for better contrast against light and dark tiles */}
      <Polyline
        positions={positions}
        pathOptions={{
          color: '#001e40',
          weight: weight + 3,
          opacity: 0.35,
          lineCap: 'round',
          lineJoin: 'round'
        }}
      />
      {/* Main dashed trajectory line */}
      <Polyline
        positions={positions}
        pathOptions={{
          color: color,
          weight: weight,
          opacity: 0.95,
          dashArray: dashArray,
          lineCap: 'round',
          lineJoin: 'round'
        }}
      />
    </>
  );
};

export default TrajectoryLine;
