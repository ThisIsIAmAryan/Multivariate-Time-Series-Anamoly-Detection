"""
Multivariate Time Series Anomaly Detection System

This module implements a comprehensive anomaly detection system for multivariate time series data using multiple detection algorithms including Isolation Forest, PCA-based detection, and statistical threshold-based methods.

Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data loading, validation, and preprocessing operations."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        
    def load_and_validate_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV data and perform basic validation.
        
        Args:
            csv_path: Path to the input CSV file
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Validate required columns
            if 'Time' not in df.columns:
                raise ValueError("Time column not found in dataset")
                
            # Convert time column to datetime
            df['Time'] = pd.to_datetime(df['Time'])
            
            # Identify feature columns (all except Time)
            self.feature_columns = [col for col in df.columns if col != 'Time']
            logger.info(f"Identified {len(self.feature_columns)} feature columns")
            
            # Check for minimum data requirements
            if len(df) < 72 * 60:  # Minimum 72 hours of minute-level data
                logger.warning(f"Dataset has only {len(df)} rows, which may be insufficient for training")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward-fill and interpolation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Check for missing values
        missing_counts = df_processed[self.feature_columns].isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values in {missing_counts[missing_counts > 0].shape[0]} columns")
            
            # Forward fill then backward fill
            df_processed[self.feature_columns] = df_processed[self.feature_columns].fillna(method='ffill')
            df_processed[self.feature_columns] = df_processed[self.feature_columns].fillna(method='bfill')
            
            # Linear interpolation for any remaining missing values
            df_processed[self.feature_columns] = df_processed[self.feature_columns].interpolate(method='linear')
        
        return df_processed
    
    def split_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training period (normal data) and full analysis period.
        
        Training period: 1/1/2004 0:00 to 1/5/2004 23:59 (120 hours)
        Analysis period: 1/1/2004 0:00 to 1/19/2004 7:59 (439 hours)
        
        Args:
            df: Input DataFrame with Time column
            
        Returns:
            Tuple of (training_data, full_data)
        """
        # Define time ranges
        train_start = pd.to_datetime('2004-01-01 00:00:00')
        train_end = pd.to_datetime('2004-01-05 23:59:59')
        
        # Filter training data
        train_mask = (df['Time'] >= train_start) & (df['Time'] <= train_end)
        training_data = df[train_mask].copy()
        
        logger.info(f"Training data: {len(training_data)} samples from {training_data['Time'].min()} to {training_data['Time'].max()}")
        logger.info(f"Full analysis data: {len(df)} samples from {df['Time'].min()} to {df['Time'].max()}")
        
        if len(training_data) < 100:
            raise ValueError(f"Insufficient training data: only {len(training_data)} samples")
        
        return training_data, df
    
    def scale_features(self, train_data: pd.DataFrame, full_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler fitted on training data.
        
        Args:
            train_data: Training DataFrame
            full_data: Full DataFrame for analysis
            
        Returns:
            Tuple of (scaled_train_features, scaled_full_features)
        """
        # Fit scaler on training data only
        train_features = train_data[self.feature_columns].values
        self.scaler.fit(train_features)
        
        # Transform both datasets
        scaled_train = self.scaler.transform(train_features)
        scaled_full = self.scaler.transform(full_data[self.feature_columns].values)
        
        logger.info(f"Scaled features - Training: {scaled_train.shape}, Full: {scaled_full.shape}")
        
        return scaled_train, scaled_full


class AnomalyDetector:
    """Main anomaly detection class implementing multiple detection algorithms."""
    
    def __init__(self):
        self.isolation_forest = None
        self.pca_model = None
        self.threshold_stats = {}
        self.feature_columns: List[str] = []
        
    def train_models(self, train_data: np.ndarray, feature_columns: List[str]) -> None:
        """
        Train all anomaly detection models on normal data.
        
        Args:
            train_data: Scaled training data
            feature_columns: List of feature column names
        """
        self.feature_columns = feature_columns
        
        # 1. Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% outliers even in training
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(train_data)
        logger.info("Trained Isolation Forest model")
        
        # 2. Train PCA model for reconstruction error
        # Use enough components to capture 95% of variance
        self.pca_model = PCA(n_components=min(0.95, len(feature_columns)-1))
        self.pca_model.fit(train_data)
        logger.info(f"Trained PCA model with {self.pca_model.n_components_} components")
        
        # 3. Calculate statistical thresholds
        self._calculate_threshold_stats(train_data)
        
    def _calculate_threshold_stats(self, train_data: np.ndarray) -> None:
        """Calculate statistical thresholds for each feature."""
        self.threshold_stats = {}
        for i, feature in enumerate(self.feature_columns):
            feature_data = train_data[:, i]
            self.threshold_stats[feature] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q75': np.percentile(feature_data, 75)
            }
    
    def detect_anomalies(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies using multiple methods and return individual scores.
        
        Args:
            data: Scaled data for anomaly detection
            
        Returns:
            Dictionary containing scores from different methods
        """
        results = {}
        
        # 1. Isolation Forest scores
        if_scores = self.isolation_forest.decision_function(data)
        # Convert to anomaly scores (higher = more anomalous)
        results['isolation_forest'] = -if_scores
        
        # 2. PCA reconstruction error
        pca_reconstructed = self.pca_model.inverse_transform(self.pca_model.transform(data))
        reconstruction_errors = np.mean((data - pca_reconstructed) ** 2, axis=1)
        results['pca_reconstruction'] = reconstruction_errors
        
        # 3. Statistical threshold violations
        threshold_scores = self._calculate_threshold_scores(data)
        results['threshold_violation'] = threshold_scores
        
        logger.info("Calculated anomaly scores using all methods")
        return results
    
    def _calculate_threshold_scores(self, data: np.ndarray) -> np.ndarray:
        """Calculate threshold violation scores for each sample."""
        scores = np.zeros(len(data))
        
        for i, feature in enumerate(self.feature_columns):
            feature_data = data[:, i]
            stats = self.threshold_stats[feature]
            
            # Calculate z-scores
            z_scores = np.abs((feature_data - stats['mean']) / stats['std'])
            
            # Score based on standard deviations from mean
            feature_scores = np.where(z_scores > 3, z_scores, 0)
            scores += feature_scores
        
        return scores
    
    def calculate_feature_contributions(self, data: np.ndarray, method_scores: Dict[str, np.ndarray]) -> List[List[str]]:
        """
        Calculate top contributing features for each anomaly.
        
        Args:
            data: Scaled data
            method_scores: Scores from different detection methods
            
        Returns:
            List of lists containing top contributing features for each sample
        """
        n_samples = len(data)
        contributions = []
        
        for i in range(n_samples):
            sample_contributions = self._calculate_sample_contributions(data[i], method_scores, i)
            contributions.append(sample_contributions)
        
        return contributions
    
    def _calculate_sample_contributions(self, sample: np.ndarray, method_scores: Dict[str, np.ndarray], 
                                     sample_idx: int) -> List[str]:
        """Calculate feature contributions for a single sample."""
        feature_scores = {}
        
        # 1. Isolation Forest feature importance (approximated)
        if_score = method_scores['isolation_forest'][sample_idx]
        
        # 2. PCA contribution scores
        pca_transformed = self.pca_model.transform(sample.reshape(1, -1))
        pca_reconstructed = self.pca_model.inverse_transform(pca_transformed)
        reconstruction_errors = (sample - pca_reconstructed.flatten()) ** 2
        
        # 3. Threshold violation scores per feature
        for j, feature in enumerate(self.feature_columns):
            stats = self.threshold_stats[feature]
            z_score = abs((sample[j] - stats['mean']) / stats['std'])
            
            # Combine different contribution measures
            combined_score = (
                reconstruction_errors[j] * 0.4 +  # PCA reconstruction error
                max(0, z_score - 2) * 0.4 +       # Threshold violation
                abs(sample[j]) * 0.2               # Magnitude contribution
            )
            
            feature_scores[feature] = combined_score
        
        # Sort features by contribution and return top 7
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Only include features that actually contribute (score > threshold)
        contributing_features = [feat for feat, score in sorted_features if score > 0.1][:7]
        
        # Fill remaining slots with empty strings if fewer than 7 contributors
        while len(contributing_features) < 7:
            contributing_features.append("")
        
        return contributing_features


class AnomalyScorer:
    """Handles conversion of raw anomaly scores to 0-100 scale."""
    
    def __init__(self):
        self.score_percentiles = {}
    
    def fit_score_distribution(self, method_scores: Dict[str, np.ndarray], 
                             training_indices: np.ndarray) -> None:
        """
        Fit score distribution based on training data to establish percentiles.
        
        Args:
            method_scores: Raw scores from different methods
            training_indices: Boolean array indicating training samples
        """
        self.score_percentiles = {}
        
        for method, scores in method_scores.items():
            train_scores = scores[training_indices]
            
            self.score_percentiles[method] = {
                'p90': np.percentile(train_scores, 90),
                'p95': np.percentile(train_scores, 95),
                'p99': np.percentile(train_scores, 99),
                'max': np.max(train_scores)
            }
        
        logger.info("Fitted score distribution for normalization")
    
    def normalize_scores(self, method_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert raw scores to 0-100 scale using ensemble approach.
        
        Args:
            method_scores: Raw scores from different methods
            
        Returns:
            Normalized scores on 0-100 scale
        """
        normalized_methods = {}
        
        # Normalize each method's scores
        for method, scores in method_scores.items():
            percentiles = self.score_percentiles[method]
            
            # Map scores to 0-100 scale
            normalized = np.zeros_like(scores)
            
            # 0-10: Below 90th percentile of training data
            mask_normal = scores <= percentiles['p90']
            normalized[mask_normal] = (scores[mask_normal] / percentiles['p90']) * 10
            
            # 11-30: 90th to 95th percentile
            mask_slight = (scores > percentiles['p90']) & (scores <= percentiles['p95'])
            normalized[mask_slight] = 10 + ((scores[mask_slight] - percentiles['p90']) / 
                                           (percentiles['p95'] - percentiles['p90'])) * 20
            
            # 31-60: 95th to 99th percentile
            mask_moderate = (scores > percentiles['p95']) & (scores <= percentiles['p99'])
            normalized[mask_moderate] = 30 + ((scores[mask_moderate] - percentiles['p95']) / 
                                             (percentiles['p99'] - percentiles['p95'])) * 30
            
            # 61-90: 99th percentile to max training
            mask_significant = (scores > percentiles['p99']) & (scores <= percentiles['max'])
            if percentiles['max'] > percentiles['p99']:
                normalized[mask_significant] = 60 + ((scores[mask_significant] - percentiles['p99']) / 
                                                   (percentiles['max'] - percentiles['p99'])) * 30
            
            # 91-100: Beyond max training score
            mask_severe = scores > percentiles['max']
            normalized[mask_severe] = 90 + np.minimum((scores[mask_severe] - percentiles['max']) / 
                                                    percentiles['max'] * 10, 10)
            
            normalized_methods[method] = normalized
        
        # Ensemble: weighted average of normalized scores
        ensemble_scores = (
            normalized_methods['isolation_forest'] * 0.4 +
            normalized_methods['pca_reconstruction'] * 0.35 +
            normalized_methods['threshold_violation'] * 0.25
        )
        
        # Ensure scores are in valid range
        ensemble_scores = np.clip(ensemble_scores, 0, 100)
        
        return ensemble_scores


class TimeSeriesAnomalyDetection:
    """Main class orchestrating the entire anomaly detection pipeline."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.detector = AnomalyDetector()
        self.scorer = AnomalyScorer()
        
    def detect_anomalies(self, input_csv_path: str, output_csv_path: str) -> None:
        """
        Main method to detect anomalies and generate output CSV.
        
        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path for output CSV file
        """
        try:
            logger.info(f"Starting anomaly detection for {input_csv_path}")
            
            # 1. Load and preprocess data
            df = self.preprocessor.load_and_validate_data(input_csv_path)
            df = self.preprocessor.handle_missing_values(df)
            
            # 2. Split training and analysis data
            train_df, full_df = self.preprocessor.split_training_data(df)
            
            # 3. Scale features
            scaled_train, scaled_full = self.preprocessor.scale_features(train_df, full_df)
            
            # 4. Train models
            self.detector.train_models(scaled_train, self.preprocessor.feature_columns)
            
            # 5. Detect anomalies
            method_scores = self.detector.detect_anomalies(scaled_full)
            
            # 6. Identify training samples for score normalization
            train_start = pd.to_datetime('2004-01-01 00:00:00')
            train_end = pd.to_datetime('2004-01-05 23:59:59')
            training_mask = (full_df['Time'] >= train_start) & (full_df['Time'] <= train_end)
            
            # 7. Normalize scores
            self.scorer.fit_score_distribution(method_scores, training_mask.values)
            final_scores = self.scorer.normalize_scores(method_scores)
            
            # 8. Calculate feature contributions
            feature_contributions = self.detector.calculate_feature_contributions(scaled_full, method_scores)
            
            # 9. Generate output
            self._generate_output(full_df, final_scores, feature_contributions, output_csv_path)
            
            # 10. Validate results
            self._validate_results(final_scores, training_mask.values)
            
            logger.info(f"Anomaly detection completed. Output saved to {output_csv_path}")
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    def _generate_output(self, df: pd.DataFrame, scores: np.ndarray, 
                        contributions: List[List[str]], output_path: str) -> None:
        """Generate output CSV with anomaly scores and top contributing features."""
        
        # Create output dataframe
        output_df = df.copy()
        
        # Add abnormality score
        output_df['abnormality_score'] = scores.round(2)
        
        # Add top contributing features
        for i in range(7):
            feature_col = f'top_feature_{i+1}'
            output_df[feature_col] = [contrib[i] if i < len(contrib) else "" for contrib in contributions]
        
        # Save output
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"Generated output with {len(output_df)} rows and {len(output_df.columns)} columns")
    
    def _validate_results(self, scores: np.ndarray, training_mask: np.ndarray) -> None:
        """Validate the results meet success criteria."""
        
        # Check training period scores
        training_scores = scores[training_mask]
        train_mean = np.mean(training_scores)
        train_max = np.max(training_scores)
        
        logger.info(f"Training period validation:")
        logger.info(f"  Mean score: {train_mean:.2f} (should be < 10)")
        logger.info(f"  Max score: {train_max:.2f} (should be < 25)")
        
        if train_mean >= 10:
            logger.warning(f"Training period mean score ({train_mean:.2f}) is >= 10")
        
        if train_max >= 25:
            logger.warning(f"Training period max score ({train_max:.2f}) is >= 25")
        
        # Check for sudden score jumps
        score_diffs = np.abs(np.diff(scores))
        max_jump = np.max(score_diffs)
        mean_jump = np.mean(score_diffs)
        
        logger.info(f"Score stability:")
        logger.info(f"  Max score jump: {max_jump:.2f}")
        logger.info(f"  Mean score jump: {mean_jump:.2f}")
        
        # Overall statistics
        logger.info(f"Overall score statistics:")
        logger.info(f"  Min: {np.min(scores):.2f}")
        logger.info(f"  Max: {np.max(scores):.2f}")
        logger.info(f"  Mean: {np.mean(scores):.2f}")
        logger.info(f"  Std: {np.std(scores):.2f}")


def main(input_csv_path: str, output_csv_path: str) -> None:
    """
    Main function to run anomaly detection.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path for output CSV file
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Create and run anomaly detection
    detector = TimeSeriesAnomalyDetection()
    detector.detect_anomalies(input_csv_path, output_csv_path)


if __name__ == "__main__":
    # Example usage
    input_file = "81ce1f00-c3f4-4baa-9b57-006fad1875adTEP_Train_Test.csv"
    output_file = "anomaly_detection_results.csv"
    
    main(input_file, output_file)
