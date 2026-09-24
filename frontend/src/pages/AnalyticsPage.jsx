import React, { useState, useEffect } from 'react';
import { analyticsData } from '../data/mockData';
import { getTrafficAnalytics } from '../services/gisService';

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('Today (Real-time)');
  const [realAnalytics, setRealAnalytics] = useState(null);

  useEffect(() => {
    let isMounted = true;
    getTrafficAnalytics().then(data => {
      if (isMounted && data) {
        setRealAnalytics(data);
      }
    });
    return () => { isMounted = false; };
  }, []);

  const maxHourly = Math.max(...analyticsData.hourlyVolume.map(h => h.count));

  // Compute real vehicle class breakdown if available from AI Engine
  let displayClasses = analyticsData.vehicleClasses;
  if (realAnalytics && realAnalytics.vehicle_class_statistics) {
    const totalVehicles = Object.values(realAnalytics.vehicle_class_statistics).reduce((a, b) => a + b, 0);
    const colorMap = {
      auto_rickshaw: '#00677d',
      bus: '#003366',
      car: '#138808',
      truck: '#ff7a1a',
      motorcycle: '#8b5cf6',
      van: '#dc2626'
    };
    displayClasses = Object.entries(realAnalytics.vehicle_class_statistics).map(([clsName, count]) => {
      const formattedName = clsName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
      const pct = totalVehicles > 0 ? Math.round((count / totalVehicles) * 100) : 0;
      return {
        name: `${formattedName} (${count})`,
        percentage: pct,
        color: colorMap[clsName] || '#003366'
      };
    });
  }

  const odData = realAnalytics?.od_statistics;


  return (
    <div className="container-7xl" style={{ paddingTop: '1.75rem', paddingBottom: '3rem' }}>
      {/* Page Title & Filter Bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1.5rem',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div>
          <span style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#00677d' }}>
            Telemetry Intelligence
          </span>
          <h1 style={{ fontSize: '24px', fontWeight: 800, color: 'var(--primary-container, #003366)', letterSpacing: '-0.02em' }}>
            Traffic Flow &amp; Incident Analytics
          </h1>
          <p style={{ fontSize: '13px', color: '#64748b' }}>
            Deep-dive predictive analytics across metropolitan grid sectors
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {['Today (Real-time)', 'Last 7 Days', 'Monthly SIH Audit'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              style={{
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: 600,
                borderRadius: '8px',
                border: '1px solid #cbd5e1',
                backgroundColor: timeRange === range ? '#003366' : '#ffffff',
                color: timeRange === range ? '#ffffff' : '#475569',
                cursor: 'pointer'
              }}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Hourly Volume + Vehicle Distribution */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
        gap: '1.5rem',
        marginBottom: '1.5rem'
      }}>
        {/* Hourly Traffic Volume Bar Chart */}
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          padding: '1.5rem',
          border: '1px solid #e2e8f0',
          boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#001e40' }}>Diurnal Traffic Flow Distribution</h3>
              <p style={{ fontSize: '12px', color: '#64748b' }}>Hourly vehicle ingress rate across Coimbatore urban corridors</p>
            </div>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#0284c7', backgroundColor: '#e0f2fe', padding: '2px 8px', borderRadius: '4px' }}>
              Peak 18:00 (395k/hr)
            </span>
          </div>

          {/* Simulated CSS Bar Chart */}
          <div style={{
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
            height: '200px',
            paddingTop: '1rem',
            borderBottom: '1px solid #cbd5e1'
          }}>
            {analyticsData.hourlyVolume.map((item) => {
              const heightPct = Math.round((item.count / maxHourly) * 100);
              const isPeak = item.count === maxHourly;
              return (
                <div key={item.hour} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
                  <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px', fontWeight: 600 }}>
                    {(item.count / 1000).toFixed(0)}k
                  </div>
                  <div
                    style={{
                      width: '28px',
                      height: `${heightPct}%`,
                      backgroundColor: isPeak ? '#ff7a1a' : '#003366',
                      borderRadius: '4px 4px 0 0',
                      transition: 'height 0.4s ease',
                      boxShadow: isPeak ? '0 2px 8px rgba(255, 122, 0, 0.4)' : 'none'
                    }}
                    title={`${item.hour}: ${item.count.toLocaleString()} vehicles`}
                  />
                  <div style={{ fontSize: '10px', color: '#475569', marginTop: '6px', fontWeight: 500 }}>
                    {item.hour}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Vehicle Classification Breakdown */}
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          padding: '1.5rem',
          border: '1px solid #e2e8f0',
          boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#001e40' }}>AI Vehicle Categorization</h3>
              <p style={{ fontSize: '12px', color: '#64748b' }}>Edge vision neural classification breakdown</p>
            </div>
            <span className="material-symbols-outlined" style={{ color: '#138808' }}>pie_chart</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {displayClasses.map((cls) => (
              <div key={cls.name}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12.5px', marginBottom: '4px' }}>
                  <span style={{ fontWeight: 600, color: '#1e293b' }}>{cls.name}</span>
                  <span style={{ fontWeight: 700, color: cls.color }}>{cls.percentage}%</span>
                </div>
                <div style={{ width: '100%', height: '7px', backgroundColor: '#f1f5f9', borderRadius: '9999px', overflow: 'hidden' }}>
                  <div style={{ width: `${cls.percentage}%`, height: '100%', backgroundColor: cls.color, borderRadius: '9999px' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Real AI Origin-Destination (OD) Corridor Flow Matrix */}
      {odData && odData.cameras && (
        <div style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          padding: '1.5rem',
          border: '1px solid #e2e8f0',
          boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)',
          marginBottom: '1.5rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#001e40' }}>
                Multi-Camera Origin-Destination (OD) Flow Matrix
              </h3>
              <p style={{ fontSize: '12px', color: '#64748b' }}>
                Reconstructed corridor vehicle transitions ({odData.total_trips} trips, {odData.multi_camera_vehicles} cross-camera vehicles)
              </p>
            </div>
            <span style={{
              padding: '3px 8px',
              backgroundColor: '#ecfdf5',
              color: '#065f46',
              border: '1px solid #a7f3d0',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: 700
            }}>
              Real AI Transition Graph
            </span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ backgroundColor: '#f8fafc', borderBottom: '2px solid #cbd5e1' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 700, color: '#475569' }}>Origin \ Dest</th>
                  {odData.cameras.map((c) => (
                    <th key={c} style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: '#003366' }}>
                      {c}
                    </th>
                  ))}
                  <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: '#00677d' }}>Total Departures</th>
                </tr>
              </thead>
              <tbody>
                {odData.cameras.map((orig) => (
                  <tr key={orig} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{orig}</td>
                    {odData.cameras.map((dest) => {
                      const count = odData.matrix[orig]?.[dest] || 0;
                      return (
                        <td
                          key={dest}
                          style={{
                            padding: '8px 12px',
                            textAlign: 'center',
                            fontWeight: count > 0 ? 700 : 400,
                            color: count > 0 ? '#ff7a1a' : '#94a3b8',
                            backgroundColor: count > 0 ? '#fff7ed' : 'transparent',
                            borderRadius: '4px'
                          }}
                        >
                          {count}
                        </td>
                      );
                    })}
                    <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: '#00677d' }}>
                      {odData.origin_counts?.[orig] || 0}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Violations Summary Table */}
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '16px',
        padding: '1.5rem',
        border: '1px solid #e2e8f0',
        boxShadow: '0 2px 8px rgba(0, 35, 80, 0.04)'
      }}>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#001e40' }}>Live Enforcement &amp; Infraction Telemetry</h3>
            <p style={{ fontSize: '12px', color: '#64748b' }}>Automated e-Challan generation pipeline</p>
          </div>
          <span style={{
            padding: '3px 8px',
            backgroundColor: '#fef2f2',
            color: '#991b1b',
            border: '1px solid #fecaca',
            borderRadius: '6px',
            fontSize: '11px',
            fontWeight: 700
          }}>
            Automated e-Challan V-3.1
          </span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '1rem'
        }}>
          {analyticsData.violationsToday.map((v) => (
            <div
              key={v.type}
              style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '10px',
                padding: '1rem',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between'
              }}
            >
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '0.5rem' }}>
                {v.type}
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '24px', fontWeight: 800, color: '#dc2626' }}>{v.count}</span>
                <span style={{ fontSize: '12px', fontWeight: 700, color: v.trend.startsWith('-') ? '#16a34a' : '#ea580c' }}>
                  {v.trend} today
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
