# Anomaly Detection Frontend

A modern, interactive dashboard for visualizing multivariate time series anomaly detection results.

## Features

### 📊 **Comprehensive Visualizations**
- **Time Series Chart**: Interactive line chart showing anomaly scores over time
- **Severity Distribution**: Pie chart breaking down anomalies by severity levels
- **Feature Contributions**: Bar chart of most contributing features
- **Score Distribution**: Area chart showing score patterns
- **Real-time Statistics**: Live metrics cards with key insights

### 🎨 **Modern UI/UX**
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Dark Mode Support**: Toggle between light and dark themes
- **Interactive Filters**: Filter by severity, time range, and other criteria
- **Professional Layout**: Clean, industrial-grade dashboard design

### 🔧 **Technical Features**
- **CSV Upload**: Drag-and-drop or click to upload anomaly detection results
- **Data Export**: Export filtered results and visualizations
- **Real-time Updates**: Live data refresh and filtering
- **TypeScript**: Full type safety and better development experience

## Technology Stack

- **Framework**: Next.js 15.5 with React 19
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts (responsive chart library)
- **Icons**: Lucide React (modern icon library)
- **Language**: TypeScript
- **Data Processing**: CSV parsing and manipulation

## Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn package manager

### Installation

1. **Navigate to frontend directory**:
   ```bash
   cd anomaly-detection-frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open browser**:
   Navigate to `http://localhost:3000`

### Building for Production

```bash
npm run build
npm start
```

## Dashboard Components

The dashboard includes comprehensive visualizations for anomaly detection results:

- Interactive time series charts
- Severity distribution pie charts
- Feature contribution analysis
- Real-time statistics and filtering
- Anomaly details table with color-coded severity levels

## Integration

This frontend is designed to work with the Python backend anomaly detection system. Upload your `anomaly_detection_results.csv` file to visualize the results.
