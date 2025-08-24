#!/bin/bash
# Setup script for the Anomaly Detection Frontend

echo "🚀 Setting up Anomaly Detection Frontend..."
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the frontend directory."
    exit 1
fi

echo "📦 Installing dependencies..."
npm install

echo ""
echo "📊 Installing visualization libraries..."
npm install recharts lucide-react papaparse date-fns clsx

echo ""
echo "🔧 Installing development dependencies..."
npm install --save-dev @types/papaparse

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Run 'npm run dev' to start the development server"
echo "  2. Open http://localhost:3000 in your browser"
echo "  3. Upload your anomaly_detection_results.csv file"
echo ""
echo "📁 Project structure:"
echo "  ├── src/app/page.tsx          # Main dashboard component"
echo "  ├── src/components/           # Reusable components"
echo "  ├── package.json              # Dependencies"
echo "  └── README.md                 # Documentation"
echo ""
echo "🎨 Features available after setup:"
echo "  ✓ Interactive time series charts"
echo "  ✓ Severity distribution pie charts"
echo "  ✓ Feature contribution analysis"
echo "  ✓ Real-time filtering and search"
echo "  ✓ CSV upload and data processing"
echo "  ✓ Export functionality"
echo "  ✓ Responsive design"
echo "  ✓ Dark mode support"
echo ""
echo "Happy analyzing! 📈"
