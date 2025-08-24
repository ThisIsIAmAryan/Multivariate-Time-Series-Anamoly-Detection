# Vercel Deployment Guide

## 🚀 Deploy Your Anomaly Detection Dashboard to Vercel

Follow these steps to deploy your interactive dashboard to Vercel:

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Visit Vercel**: Go to [vercel.com](https://vercel.com)
2. **Sign in with GitHub**: Use your GitHub account
3. **Import Project**: 
   - Click "New Project"
   - Select your repository: `Multivariate-Time-Series-Anamoly-Detection`
4. **Configure Project**:
   - **Framework Preset**: Next.js
   - **Root Directory**: `anomaly-detection-frontend`
   - **Build Command**: `npm run build`
   - **Install Command**: `npm install --legacy-peer-deps`
5. **Deploy**: Click "Deploy" and wait for completion

### Option 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to frontend directory
cd anomaly-detection-frontend

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

### Configuration Details

#### Root Directory
Set **Root Directory** to: `anomaly-detection-frontend`

#### Build Settings
- **Framework**: Next.js
- **Build Command**: `npm run build` 
- **Install Command**: `npm install --legacy-peer-deps`
- **Output Directory**: `.next` (auto-detected)

#### Environment Variables
No environment variables required for this deployment.

### Expected Results

✅ **Successful Deployment**:
- Interactive anomaly detection dashboard
- Real-time charts and visualizations  
- Search and filtering capabilities
- Professional UI with 26,400+ data samples

✅ **Performance**:
- Static site generation for fast loading
- Optimized React components
- Responsive design for all devices

### Troubleshooting

#### Common Issues:

1. **Build Errors**: 
   - Ensure TypeScript errors are resolved
   - Check ESLint configuration

2. **Install Issues**:
   - Use `--legacy-peer-deps` flag for React version compatibility

3. **Root Directory**:
   - Make sure to set root directory to `anomaly-detection-frontend`

#### Support
- Check build logs in Vercel dashboard
- Verify GitHub repository is up to date
- Ensure all files are committed

### Expected URL Structure
Your deployed app will be available at:
`https://multivariate-time-series-anamoly-detection-[hash].vercel.app`

---
🎉 **Your anomaly detection dashboard will be live and accessible worldwide!**
