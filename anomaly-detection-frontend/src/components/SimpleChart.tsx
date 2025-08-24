// components/SimpleChart.tsx
'use client';

import React from 'react';

interface DataPoint {
  Time: string;
  abnormality_score: number;
}

interface SimpleChartProps {
  data: DataPoint[];
  width?: number;
  height?: number;
}

export function SimpleLineChart({ data, width = 600, height = 300 }: SimpleChartProps) {
  if (!data || data.length === 0) return <div>No data available</div>;

  const maxScore = Math.max(...data.map(d => d.abnormality_score));
  const minScore = Math.min(...data.map(d => d.abnormality_score));
  const scoreRange = maxScore - minScore || 1;

  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * (width - 40) + 20;
    const y = height - 20 - ((d.abnormality_score - minScore) / scoreRange) * (height - 40);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Anomaly Score Over Time</h3>
      <svg width={width} height={height} className="border">
        {/* Y-axis */}
        <line x1="20" y1="20" x2="20" y2={height - 20} stroke="#666" strokeWidth="1" />
        {/* X-axis */}
        <line x1="20" y1={height - 20} x2={width - 20} y2={height - 20} stroke="#666" strokeWidth="1" />
        
        {/* Y-axis labels */}
        {[0, 25, 50, 75, 100].map(val => {
          const y = height - 20 - (val / 100) * (height - 40);
          return (
            <g key={val}>
              <line x1="15" y1={y} x2="25" y2={y} stroke="#666" strokeWidth="1" />
              <text x="10" y={y + 4} textAnchor="end" fontSize="12" fill="#666">{val}</text>
            </g>
          );
        })}
        
        {/* Data line */}
        <polyline
          points={points}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
        />
        
        {/* Data points */}
        {data.map((d, i) => {
          const x = (i / (data.length - 1)) * (width - 40) + 20;
          const y = height - 20 - ((d.abnormality_score - minScore) / scoreRange) * (height - 40);
          
          let color = '#22c55e'; // green
          if (d.abnormality_score > 30) color = '#f97316'; // orange
          if (d.abnormality_score > 60) color = '#ef4444'; // red
          if (d.abnormality_score > 90) color = '#991b1b'; // dark red
          
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r="3"
              fill={color}
              className="hover:r-5 transition-all cursor-pointer"
            >
              <title>{`${d.Time}: ${d.abnormality_score.toFixed(2)}`}</title>
            </circle>
          );
        })}
      </svg>
    </div>
  );
}

export function SimplePieChart({ data }: { data: { name: string; value: number; color: string }[] }) {
  const total = data.reduce((sum, d) => sum + d.value, 0);
  let currentAngle = 0;
  const radius = 80;
  const centerX = 100;
  const centerY = 100;

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Severity Distribution</h3>
      <div className="flex items-center">
        <svg width="200" height="200">
          {data.map((segment, i) => {
            const angle = (segment.value / total) * 360;
            const startAngle = currentAngle;
            const endAngle = currentAngle + angle;
            
            const x1 = centerX + radius * Math.cos((startAngle - 90) * Math.PI / 180);
            const y1 = centerY + radius * Math.sin((startAngle - 90) * Math.PI / 180);
            const x2 = centerX + radius * Math.cos((endAngle - 90) * Math.PI / 180);
            const y2 = centerY + radius * Math.sin((endAngle - 90) * Math.PI / 180);
            
            const largeArcFlag = angle > 180 ? 1 : 0;
            
            const pathData = [
              `M ${centerX} ${centerY}`,
              `L ${x1} ${y1}`,
              `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
              'Z'
            ].join(' ');
            
            currentAngle += angle;
            
            return (
              <path
                key={i}
                d={pathData}
                fill={segment.color}
                stroke="white"
                strokeWidth="2"
                className="hover:opacity-80 cursor-pointer"
              >
                <title>{`${segment.name}: ${segment.value} (${(segment.value/total*100).toFixed(1)}%)`}</title>
              </path>
            );
          })}
        </svg>
        
        <div className="ml-4">
          {data.map((segment, i) => (
            <div key={i} className="flex items-center mb-2">
              <div 
                className="w-4 h-4 rounded mr-2" 
                style={{ backgroundColor: segment.color }}
              />
              <span className="text-sm">
                {segment.name}: {segment.value} ({(segment.value/total*100).toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
