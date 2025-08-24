# Multivariate Time Series Anomaly Detection

🚀 **Advanced machine learning solution for detecting anomalies in multivariate time series data using ensemble methods**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org)
[![Next.js](https://img.shields.io/badge/Next.js-15.5-black.svg)](https://nextjs.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org)

## 🎯 Overview

This project implements a comprehensive anomaly detection system for multivariate time series data, specifically designed for industrial process monitoring. Using the Tennessee Eastman Process (TEP) dataset, the system combines multiple machine learning algorithms to detect anomalies and identify their root causes.

### ✨ Key Features

- 🤖 **Ensemble Anomaly Detection**: Combines Isolation Forest, PCA, and statistical methods
- 📊 **Feature Attribution**: Identifies top 7 contributing features for each anomaly
- 🎚️ **Severity Scoring**: 0-100 anomaly scores with 5-level severity classification
- 📈 **Interactive Dashboard**: Modern React-based visualization interface
- ⏱️ **Time-Aware Validation**: Proper temporal data splitting for realistic evaluation
- 🔍 **Real-Time Analysis**: Processes 26,400 samples across 52 features

## 🏗️ Architecture

```
📦 Multivariate Time Series Anomaly Detection
├── 🐍 anomaly_detection.py          # Main ML pipeline
├── 📊 anomaly_detection_results.csv # Output results (26,400 samples)
├── 📋 summary.py                    # Analysis & statistics
├── ✅ validation_comparison.py      # Model validation
├── 📚 PROJECT_DOCUMENTATION.md     # Complete documentation
├── 📝 SOLUTION_OUTLINE.md          # Technical approach
└── 🌐 anomaly-detection-frontend/   # Interactive dashboard
    ├── src/app/page.tsx             # Main dashboard
    ├── src/components/              # React components
    └── public/                      # Static assets
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThisIsIAmAryan/Multivariate-Time-Series-Anamoly-Detection.git
   cd "Multivariate-Time-Series-Anamoly-Detection"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install scikit-learn pandas numpy matplotlib
   ```

4. **Run anomaly detection**
   ```bash
   python anomaly_detection.py
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd anomaly-detection-frontend
   ```

2. **Install dependencies**
   ```bash
   npm install --legacy-peer-deps
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Open dashboard**
   - Visit: http://localhost:3000
   - Interactive charts and real-time data exploration

## 🔬 Technical Approach

### Machine Learning Pipeline

#### 1. **Data Preprocessing**
- Tennessee Eastman Process dataset (52 features, 26,400 samples)
- Time-aware splitting: 27.3% training, 72.7% analysis
- StandardScaler normalization

#### 2. **Ensemble Anomaly Detection**
- **Isolation Forest**: Tree-based outlier detection
- **PCA Reconstruction**: Dimensionality-based anomaly scoring
- **Statistical Analysis**: Z-score and percentile thresholds

#### 3. **Feature Attribution**
- Systematic ranking of feature contributions
- Top 7 feature identification for root cause analysis
- Importance weighting and scoring

#### 4. **Anomaly Scoring**
- Ensemble score aggregation
- 0-100 scale normalization
- Severity classification:
  - 🟢 Normal (0-10)
  - 🟡 Slight (11-30)
  - 🟠 Moderate (31-60)
  - 🔴 Significant (61-90)
  - ⚫ Severe (91-100)

## 📊 Results & Performance

### Model Performance
- ✅ **Training Performance**: Mean anomaly score 1.97 (target: <10)
- ✅ **Data Coverage**: 100% sample processing with feature attribution
- ✅ **Score Distribution**: Full 0-100 range with proper severity distribution
- ✅ **Processing Speed**: Real-time analysis capability

### Output Format
```csv
Time,abnormality_score,top_feature_1,top_feature_2,top_feature_3,top_feature_4,top_feature_5,top_feature_6,top_feature_7,[52 additional features]
```

## 🌐 Interactive Dashboard

### Features
- 📈 **Real-time Charts**: Interactive line charts with hover tooltips
- 🔍 **Search & Filter**: Advanced data exploration by time, features, scores
- 📊 **Statistics Cards**: Live metrics and severity distribution
- 🎨 **Professional UI**: Modern design with consistent styling
- 📱 **Responsive**: Works on desktop and mobile devices

### Screenshots
*Dashboard displays real anomaly detection results with interactive visualizations*

## 📁 Project Structure

```
├── Backend (Python)
│   ├── anomaly_detection.py       # Main ML pipeline (527 lines)
│   ├── summary.py                 # Enhanced analysis
│   └── validation_comparison.py   # Model validation
├── Frontend (React/Next.js)
│   ├── src/app/page.tsx           # Main dashboard (500+ lines)
│   ├── src/components/            # Reusable components
│   └── public/                    # Static assets & data
├── Data
│   ├── 81ce1f00-c3f4-4baa-9b57-006fad1875adTEP_Train_Test.csv  # Input dataset
│   └── anomaly_detection_results.csv                           # ML results
└── Documentation
    ├── PROJECT_DOCUMENTATION.md   # Complete project docs
    ├── SOLUTION_OUTLINE.md       # Technical approach
    └── README.md                 # This file
```

## 🛠️ Technologies Used

### Backend
- **Python 3.12**: Core programming language
- **Scikit-Learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization

### Frontend
- **Next.js 15.5**: React framework
- **React 18**: UI library
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS 4**: Utility-first CSS
- **Recharts**: Interactive charts
- **Lucide React**: Professional icons

## 📈 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Total Samples Processed | 26,400 | ✅ |
| Features Analyzed | 52 | ✅ |
| Mean Training Score | 1.97 | ✅ (<10 target) |
| Anomaly Score Range | 0.0-100.0 | ✅ |
| Feature Attribution Coverage | 100% | ✅ |
| Processing Time | Real-time | ✅ |

## 🔮 Future Enhancements

- 🚀 **Real-time API**: FastAPI backend for live inference
- 📡 **Streaming Data**: Integration with real-time data sources
- 🧠 **Deep Learning**: Advanced neural network approaches
- 🔔 **Alert System**: Automated notifications for critical anomalies
- 📊 **Advanced Analytics**: Trend analysis and pattern recognition

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Aryan Sharma**
- GitHub: [@ThisIsIAmAryan](https://github.com/ThisIsIAmAryan)
- Project: [Multivariate Time Series Anomaly Detection](https://github.com/ThisIsIAmAryan/Multivariate-Time-Series-Anamoly-Detection)

## 🙏 Acknowledgments

- Tennessee Eastman Process dataset for benchmarking
- Scikit-Learn community for excellent ML tools
- React and Next.js teams for modern web frameworks

---

⭐ **Star this repository if you find it helpful!**
