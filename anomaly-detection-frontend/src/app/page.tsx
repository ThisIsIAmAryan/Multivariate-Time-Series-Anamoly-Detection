'use client';

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Upload, Search, Filter, AlertTriangle, TrendingUp, Clock, Target, RefreshCw, Download } from 'lucide-react';
import Papa from 'papaparse';

// Types
interface AnomalyData {
  Time: string;
  abnormality_score: number;
  top_feature_1: string;
  top_feature_2: string;
  top_feature_3: string;
  [key: string]: any;
}

interface SeverityStats {
  normal: number;
  slight: number;
  moderate: number;
  significant: number;
  severe: number;
}

// Interactive Chart Components using Recharts
function InteractiveLineChart({ data }: { data: AnomalyData[] }) {
  if (!data || data.length === 0) return <div>No data available</div>;

  const chartData = data.slice(0, 200).map((d, i) => ({
    index: i,
    time: d.Time,
    score: d.abnormality_score,
    severity: d.abnormality_score > 75 ? 'Severe' : 
              d.abnormality_score > 50 ? 'Significant' : 
              d.abnormality_score > 25 ? 'Moderate' : 'Normal'
  }));

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4 flex items-center text-gray-900">
        <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
        Anomaly Score Over Time
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="index" 
            tick={{ fontSize: 12 }}
            label={{ value: 'Time Index', position: 'insideBottom', offset: -5 }}
          />
          <YAxis 
            domain={[0, 100]}
            tick={{ fontSize: 12 }}
            label={{ value: 'Anomaly Score', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            formatter={(value: any, name: string) => [value, 'Anomaly Score']}
            labelFormatter={(label: any) => `Time Index: ${label}`}
          />
          <Line 
            type="monotone" 
            dataKey="score" 
            stroke="#3b82f6" 
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#3b82f6' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function SimplePieChart({ data }: { data: { name: string; value: number; color: string }[] }) {
  const total = data.reduce((sum, d) => sum + d.value, 0);
  
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4 text-gray-900">Severity Distribution</h3>
      <div className="space-y-3">
        {data.map((segment, i) => {
          const percentage = total > 0 ? (segment.value / total * 100) : 0;
          return (
            <div key={i} className="flex items-center justify-between">
              <div className="flex items-center">
                <div 
                  className="w-4 h-4 rounded mr-3" 
                  style={{ backgroundColor: segment.color }}
                />
                <span className="text-sm font-medium text-gray-900">{segment.name}</span>
              </div>
              <div className="text-sm text-gray-600">
                {segment.value} ({percentage.toFixed(1)}%)
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Load real CSV data
const loadRealAnomalyData = async (): Promise<AnomalyData[]> => {
  try {
    const response = await fetch('/anomaly_detection_results.csv');
    if (!response.ok) {
      console.warn('Could not load real CSV data, using sample data instead');
      return generateSampleData();
    }
    
    const csvText = await response.text();
    const parsed = Papa.parse(csvText, {
      header: true,
      skipEmptyLines: true,
      transform: (value: string, field: string) => {
        // Keep Time field as string, convert others to numbers if possible
        if (field === 'Time') {
          return value;
        }
        
        // Convert numeric strings to numbers for other fields
        const num = parseFloat(value);
        return isNaN(num) ? value : num;
      }
    });

    if (parsed.errors.length > 0) {
      console.warn('CSV parsing errors:', parsed.errors);
    }

    const data: AnomalyData[] = parsed.data.map((row: any) => ({
      Time: String(row.Time || ''),
      abnormality_score: parseFloat(row.abnormality_score) || 0,
      top_feature_1: String(row.top_feature_1 || ''),
      top_feature_2: String(row.top_feature_2 || ''),
      top_feature_3: String(row.top_feature_3 || ''),
      top_feature_4: String(row.top_feature_4 || ''),
      top_feature_5: String(row.top_feature_5 || ''),
      top_feature_6: String(row.top_feature_6 || ''),
      top_feature_7: String(row.top_feature_7 || ''),
      // Include all other columns for completeness
      ...row
    }));

    console.log(`Loaded ${data.length} real anomaly detection results`);
    return data;
  } catch (error) {
    console.error('Error loading real data:', error);
    console.warn('Falling back to sample data');
    return generateSampleData();
  }
};

// Sample data generator (fallback)
const generateSampleData = (): AnomalyData[] => {
  const data: AnomalyData[] = [];
  const startDate = new Date('2004-01-01T00:00:00');
  
  for (let i = 0; i < 1000; i++) {
    const date = new Date(startDate.getTime() + i * 60 * 1000);
    const isTrainingPeriod = i < 300;
    
    let score = 0;
    if (isTrainingPeriod) {
      score = Math.random() * 10;
    } else {
      const rand = Math.random();
      if (rand < 0.6) score = Math.random() * 20;
      else if (rand < 0.8) score = 20 + Math.random() * 40;
      else score = 60 + Math.random() * 40;
    }
    
    data.push({
      Time: date.toISOString(),
      abnormality_score: Math.round(score * 100) / 100,
      top_feature_1: 'ReactorTemperatureDegC',
      top_feature_2: 'ReactorPressure',
      top_feature_3: 'FlowRate'
    });
  }
  
  return data;
};

export default function AnomalyDetectionDashboard() {
  const [data, setData] = useState<AnomalyData[]>([]);
  const [selectedSeverity, setSelectedSeverity] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    // Load real anomaly detection results on component mount
    loadRealAnomalyData().then(setData);
  }, []);
  
  // Filter data
  const filteredData = data.filter(item => {
    const score = item.abnormality_score;
    
    // Severity filter
    if (selectedSeverity === 'normal' && score > 10) return false;
    if (selectedSeverity === 'slight' && (score <= 10 || score > 30)) return false;
    if (selectedSeverity === 'moderate' && (score <= 30 || score > 60)) return false;
    if (selectedSeverity === 'significant' && (score <= 60 || score > 90)) return false;
    if (selectedSeverity === 'severe' && score <= 90) return false;
    
    // Search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      return (
        String(item.Time || '').toLowerCase().includes(searchLower) ||
        String(item.top_feature_1 || '').toLowerCase().includes(searchLower) ||
        String(item.top_feature_2 || '').toLowerCase().includes(searchLower) ||
        String(item.top_feature_3 || '').toLowerCase().includes(searchLower) ||
        score.toString().includes(searchTerm)
      );
    }
    
    return true;
  });
  
  // Calculate statistics
  const severityStats: SeverityStats = {
    normal: data.filter(d => d.abnormality_score <= 10).length,
    slight: data.filter(d => d.abnormality_score > 10 && d.abnormality_score <= 30).length,
    moderate: data.filter(d => d.abnormality_score > 30 && d.abnormality_score <= 60).length,
    significant: data.filter(d => d.abnormality_score > 60 && d.abnormality_score <= 90).length,
    severe: data.filter(d => d.abnormality_score > 90).length,
  };
  
  const pieData = [
    { name: 'Normal (0-10)', value: severityStats.normal, color: '#22c55e' },
    { name: 'Slight (11-30)', value: severityStats.slight, color: '#facc15' },
    { name: 'Moderate (31-60)', value: severityStats.moderate, color: '#f97316' },
    { name: 'Significant (61-90)', value: severityStats.significant, color: '#ef4444' },
    { name: 'Severe (91-100)', value: severityStats.severe, color: '#991b1b' },
  ];
  
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setLoading(true);
      // In real implementation, parse CSV here
      setTimeout(() => {
        setLoading(false);
        alert('CSV file processing would be implemented here');
      }, 1000);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 bg-blue-600 rounded flex items-center justify-center text-white font-bold">A</div>
              <h1 className="text-2xl font-bold text-gray-900">
                Multivariate Time Series Anomaly Detection
              </h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
                id="csv-upload"
              />
              <label
                htmlFor="csv-upload"
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 cursor-pointer"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload CSV
              </label>
              
              <button className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                <Download className="w-4 h-4 mr-2" />
                Export Results
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 bg-blue-500 rounded flex items-center justify-center text-white">
                    <Target className="w-5 h-5" />
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">Total Samples</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {data.length.toLocaleString()}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 bg-red-500 rounded flex items-center justify-center text-white">
                    <AlertTriangle className="w-5 h-5" />
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">High Anomalies</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {severityStats.severe + severityStats.significant}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 bg-green-500 rounded flex items-center justify-center text-white">
                    <TrendingUp className="w-5 h-5" />
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">Average Score</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {(data.reduce((sum, d) => sum + d.abnormality_score, 0) / data.length).toFixed(2)}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 bg-purple-500 rounded flex items-center justify-center text-white">
                    <Clock className="w-5 h-5" />
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">Time Range</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {Math.floor(data.length / 60)} hours
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white shadow rounded-lg p-6 mb-8">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center space-x-2">
              <Filter className="w-5 h-5 text-gray-600" />
              <span className="text-sm font-medium text-gray-700">Filters:</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <Search className="w-4 h-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search by features"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none text-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            
            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Severities</option>
              <option value="normal">Normal (0-10)</option>
              <option value="slight">Slight (11-30)</option>
              <option value="moderate">Moderate (31-60)</option>
              <option value="significant">Significant (61-90)</option>
              <option value="severe">Severe (91-100)</option>
            </select>
            
            <button
              onClick={() => {
                setLoading(true);
                loadRealAnomalyData().then(newData => {
                  setData(newData);
                  setLoading(false);
                });
              }}
              className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Loading...' : 'Refresh Data'}
            </button>
            
            <div className="text-sm text-gray-500">
              Showing {filteredData.length} of {data.length} samples
            </div>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <InteractiveLineChart data={filteredData} />
          <SimplePieChart data={pieData} />
        </div>

        {/* Recent Anomalies Table */}
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">
              Recent High-Severity Anomalies
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Severity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Top Features
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data
                  .filter(d => d.abnormality_score > 70)
                  .slice(0, 10)
                  .map((item, index) => {
                    const getSeverity = (score: number) => {
                      if (score <= 10) return { label: 'Normal', color: 'text-green-600' };
                      if (score <= 30) return { label: 'Slight', color: 'text-yellow-600' };
                      if (score <= 60) return { label: 'Moderate', color: 'text-orange-600' };
                      if (score <= 90) return { label: 'Significant', color: 'text-red-600' };
                      return { label: 'Severe', color: 'text-red-800' };
                    };
                    
                    const severity = getSeverity(item.abnormality_score);
                    
                    return (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {new Date(item.Time).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {item.abnormality_score.toFixed(2)}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${severity.color}`}>
                          {severity.label}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          <div className="flex flex-wrap gap-1">
                            {[item.top_feature_1, item.top_feature_2, item.top_feature_3]
                              .filter(f => f)
                              .map((feature, idx) => (
                                <span 
                                  key={idx}
                                  className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                                >
                                  {feature}
                                </span>
                              ))}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
