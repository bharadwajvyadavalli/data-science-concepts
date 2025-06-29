#!/usr/bin/env python3
"""
Data Science Interview Prep - NumPy and Pandas
==============================================
Core implementations for data manipulation and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import time


# ============================================================================
# NUMPY ESSENTIALS
# ============================================================================

def vectorization_demo():
    """
    Demonstrate the efficiency of vectorization over loops.
    Key interview point: Always prefer vectorized operations for performance.
    """
    print("=== Vectorization vs Loops Demo ===")

    # Create large array
    arr = np.random.randn(1000000)

    # Loop approach (slow)
    start = time.time()
    result_loop = []
    for x in arr:
        result_loop.append(x ** 2 + 2 * x + 1)
    loop_time = time.time() - start

    # Vectorized approach (fast)
    start = time.time()
    result_vec = arr ** 2 + 2 * arr + 1
    vec_time = time.time() - start

    print(f"Loop time: {loop_time:.4f}s")
    print(f"Vectorized time: {vec_time:.4f}s")
    print(f"Speedup: {loop_time / vec_time:.2f}x")
    print()


def broadcasting_demo():
    """
    Show NumPy broadcasting rules - critical for efficient operations.
    Broadcasting allows operations between arrays of different shapes.
    """
    print("=== Broadcasting Rules Demo ===")

    # 1D + 2D broadcasting
    A = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
    b = np.array([10, 20, 30])  # Shape: (3,)

    print("A shape:", A.shape)
    print("A:\n", A)
    print("\nb shape:", b.shape)
    print("b:", b)
    print("\nA + b (b is broadcast to match A's shape):\n", A + b)

    # 2D + 1D column vector
    c = np.array([[100], [200]])  # Shape: (2, 1)
    print("\nc shape:", c.shape)
    print("c:\n", c)
    print("\nA + c (c is broadcast to match A's shape):\n", A + c)
    print()


def custom_covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Implement covariance matrix from scratch.

    Covariance matrix shows how variables vary together.
    Cov(X,Y) = E[(X-μx)(Y-μy)]

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Covariance matrix of shape (n_features, n_features)
    """
    # Center the data (subtract mean)
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance
    n = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)

    return cov_matrix


def manual_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement PCA (Principal Component Analysis) from scratch.

    PCA finds the directions of maximum variance in the data.
    Steps:
    1. Center the data
    2. Compute covariance matrix
    3. Find eigenvalues and eigenvectors
    4. Select top components
    5. Transform data

    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of principal components to keep

    Returns:
        X_pca: Transformed data
        components: Principal component directions
        explained_variance: Variance explained by each component
    """
    # 1. Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # 2. Compute covariance matrix
    cov_matrix = custom_covariance_matrix(X)

    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Select top n_components
    components = eigenvectors[:, :n_components]

    # 6. Transform data
    X_pca = X_centered @ components

    return X_pca, components, eigenvalues[:n_components]


def advanced_indexing_demo():
    """
    Demonstrate advanced NumPy indexing techniques.
    Essential for efficient data manipulation in interviews.
    """
    print("=== Advanced Indexing Demo ===")

    # Create sample array
    arr = np.arange(20).reshape(4, 5)
    print("Original array:\n", arr)

    # Boolean indexing
    mask = arr > 10
    print("\nBoolean mask (arr > 10):\n", mask)
    print("Elements > 10:", arr[mask])

    # Fancy indexing
    rows = np.array([0, 2, 3])
    cols = np.array([1, 3, 4])
    print("\nFancy indexing - selecting specific elements:")
    print(f"Rows: {rows}, Cols: {cols}")
    print("Result:", arr[rows, cols])

    # Where with conditions
    result = np.where(arr % 2 == 0, arr, -1)
    print("\nReplace odd numbers with -1:\n", result)

    # Advanced slicing
    print("\nEvery other row, reverse columns:")
    print(arr[::2, ::-1])
    print()


# ============================================================================
# PANDAS PROFICIENCY
# ============================================================================

def advanced_pandas_operations():
    """
    Demonstrate advanced pandas operations commonly asked in interviews.
    """
    print("=== Advanced Pandas Operations ===")

    # Create sample data with multiple features
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.randn(100) * 10 + 50,
        'quantity': np.random.randint(1, 20, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })

    print("Sample data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    # Time series manipulation
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])

    # Complex groupby aggregation
    print("\n=== Complex Aggregation ===")
    agg_result = df.groupby(['category', 'month']).agg({
        'value': ['mean', 'std', 'count'],
        'quantity': ['sum', lambda x: x.quantile(0.75)]
    }).round(2)
    print(agg_result.head())

    # Custom apply function
    def calculate_metrics(group):
        return pd.Series({
            'total_value': (group['value'] * group['quantity']).sum(),
            'avg_value_per_item': group['value'].mean(),
            'volatility': group['value'].std() / group['value'].mean() if group['value'].mean() != 0 else 0,
            'dominant_region': group['region'].mode()[0] if len(group['region'].mode()) > 0 else 'Unknown'
        })

    print("\n=== Custom Metrics by Category ===")
    custom_agg = df.groupby('category').apply(calculate_metrics)
    print(custom_agg)

    # Window functions
    print("\n=== Rolling Window Analysis ===")
    df_sorted = df.sort_values('date')
    df_sorted['value_ma7'] = df_sorted.groupby('category')['value'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df_sorted['value_ewm'] = df_sorted.groupby('category')['value'].transform(
        lambda x: x.ewm(span=7, adjust=False).mean()
    )

    print(df_sorted[['date', 'category', 'value', 'value_ma7', 'value_ewm']].head(10))

    return df


def handle_missing_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Advanced missing data handling strategies.

    Returns:
        Cleaned dataframe and missing data summary
    """
    print("\n=== Missing Data Handling ===")

    # Create copy to avoid modifying original
    df_copy = df.copy()

    # Add some missing values for demonstration
    missing_indices = np.random.choice(df.index, size=20)
    df_copy.loc[missing_indices, 'value'] = np.nan

    # Missing data summary
    missing_summary = pd.DataFrame({
        'column': df_copy.columns,
        'missing_count': df_copy.isnull().sum().values,
        'missing_pct': (df_copy.isnull().sum() / len(df_copy) * 100).round(2).values
    })

    print("Missing data summary:")
    print(missing_summary[missing_summary['missing_count'] > 0])

    # Different imputation strategies
    for col in df_copy.columns:
        if df_copy[col].dtype in ['float64', 'int64']:
            if 'date' in df_copy.columns:
                # Forward fill for time series
                df_copy[col] = df_copy.groupby('category')[col].fillna(method='ffill')
                # Then backward fill any remaining
                df_copy[col] = df_copy.groupby('category')[col].fillna(method='bfill')
            else:
                # Mean imputation for numeric
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif df_copy[col].dtype == 'object':
            # Mode imputation for categorical
            mode_val = df_copy[col].mode()[0] if len(df_copy[col].mode()) > 0 else 'Unknown'
            df_copy[col] = df_copy[col].fillna(mode_val)

    return df_copy, missing_summary


# ============================================================================
# DATA CLEANING & EDA
# ============================================================================

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    This is a common interview question - implement a full cleaning pipeline.
    """
    print("\n=== Data Cleaning Pipeline ===")

    # Make a copy
    df_clean = df.copy()
    initial_shape = df_clean.shape

    # 1. Remove duplicates
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")

    # 2. Fix data types
    # Convert string numbers to numeric
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                # Try to convert to numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"Converted {col} to numeric")
            except:
                pass

    # 3. Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns

    # Impute numeric with median (more robust than mean)
    for col in numeric_cols:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)

    # Impute categorical with mode
    for col in categorical_cols:
        if len(df_clean[col].mode()) > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)

    # 4. Detect and handle outliers using IQR method
    print("\n=== Outlier Detection ===")
    outlier_summary = {}

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_summary[col] = len(outliers)

        # Cap outliers instead of removing
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    print("Outliers found per column:")
    for col, count in outlier_summary.items():
        if count > 0:
            print(f"  {col}: {count} outliers")

    # 5. Create derived features
    if 'date' in df_clean.columns:
        df_clean['year'] = pd.to_datetime(df_clean['date']).dt.year
        df_clean['quarter'] = pd.to_datetime(df_clean['date']).dt.quarter
        df_clean['day_of_year'] = pd.to_datetime(df_clean['date']).dt.dayofyear

    print(f"\nFinal shape: {df_clean.shape}")
    return df_clean


def generate_eda_report(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive EDA report programmatically.
    Shows what to look for in exploratory data analysis.
    """
    print("\n=== EDA Report Generation ===")

    report = {}

    # Basic information
    report['shape'] = df.shape
    report['columns'] = list(df.columns)
    report['dtypes'] = df.dtypes.value_counts().to_dict()
    report['missing_total'] = df.isnull().sum().sum()
    report['missing_by_column'] = df.isnull().sum().to_dict()

    # Memory usage
    report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 ** 2

    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Basic statistics
        report['numeric_summary'] = df[numeric_cols].describe().to_dict()

        # Correlation matrix
        report['correlation'] = df[numeric_cols].corr().to_dict()

        # Skewness and kurtosis
        report['skewness'] = {col: float(df[col].skew()) for col in numeric_cols}
        report['kurtosis'] = {col: float(df[col].kurtosis()) for col in numeric_cols}

        # Find highly correlated features
        corr_matrix = df[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr = [(i, j, corr_matrix.loc[i, j])
                     for i in upper_triangle.index
                     for j in upper_triangle.columns
                     if upper_triangle.loc[i, j] > 0.8]
        report['high_correlations'] = high_corr

    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        report['categorical_summary'] = {}
        for col in categorical_cols:
            report['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'top_5_values': df[col].value_counts().head(5).to_dict()
            }

    # Print summary
    print(f"Dataset shape: {report['shape']}")
    print(f"Memory usage: {report['memory_usage_mb']:.2f} MB")
    print(f"Missing values: {report['missing_total']}")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    if report.get('high_correlations'):
        print("\nHighly correlated features (>0.8):")
        for feat1, feat2, corr in report['high_correlations']:
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")

    return report


def visualize_eda(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
    """
    Create comprehensive EDA visualizations.
    Important for demonstrating data understanding in interviews.
    """
    print("\n=== Creating EDA Visualizations ===")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(numeric_cols) == 0:
        print("No numeric columns to visualize")
        return

    # 1. Distribution plots for numeric features
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            # Histogram with KDE
            axes[idx].hist(df[col].dropna(), bins=30, density=True,
                           alpha=0.7, edgecolor='black')

            # Add KDE
            df[col].dropna().plot.density(ax=axes[idx], color='red', linewidth=2)

            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Density')

            # Add mean and median lines
            mean_val = df[col].mean()
            median_val = df[col].median()
            axes[idx].axvline(mean_val, color='green', linestyle='--',
                              label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='orange', linestyle='--',
                              label=f'Median: {median_val:.2f}')
            axes[idx].legend()

    # Hide empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

    # 2. Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                    center=0, fmt='.2f', square=True, linewidths=1,
                    cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.show()

    # 3. Box plots for outlier detection
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        df[numeric_cols].boxplot(ax=ax, rot=45)
        plt.title('Box Plots - Outlier Detection', fontsize=16)
        plt.ylabel('Value')
        plt.tight_layout()
        plt.show()

    # 4. Categorical variables visualization
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        # Show relationship between categorical and numeric
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]

        plt.figure(figsize=(10, 6))
        df.boxplot(column=num_col, by=cat_col, ax=plt.gca())
        plt.title(f'{num_col} by {cat_col}')
        plt.suptitle('')  # Remove automatic title
        plt.tight_layout()
        plt.show()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive feature engineering pipeline.
    This demonstrates creativity in creating meaningful features.
    """
    print("\n=== Feature Engineering Pipeline ===")

    df_feat = df.copy()

    # 1. Numeric feature interactions
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns

    # Create polynomial features (be selective to avoid explosion)
    if len(numeric_cols) >= 2:
        col1, col2 = list(numeric_cols)[:2]
        df_feat[f'{col1}_times_{col2}'] = df_feat[col1] * df_feat[col2]
        df_feat[f'{col1}_div_{col2}'] = df_feat[col1] / (df_feat[col2] + 1e-8)
        df_feat[f'{col1}_squared'] = df_feat[col1] ** 2
        print(f"Created interaction features for {col1} and {col2}")

    # 2. Binning continuous variables
    if 'value' in numeric_cols:
        # Equal-width binning
        df_feat['value_quartile'] = pd.qcut(df_feat['value'], q=4,
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'])

        # Custom binning based on domain knowledge
        df_feat['value_category'] = pd.cut(df_feat['value'],
                                           bins=[-np.inf, 30, 50, 70, np.inf],
                                           labels=['Low', 'Medium', 'High', 'Very High'])

    # 3. Date-based features
    if 'date' in df_feat.columns:
        df_feat['date'] = pd.to_datetime(df_feat['date'])

        # Cyclical encoding for time features
        df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['date'].dt.day / 31)
        df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['date'].dt.day / 31)
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['date'].dt.month / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['date'].dt.month / 12)

        # Is it a special period?
        df_feat['is_quarter_end'] = df_feat['date'].dt.is_quarter_end
        df_feat['is_month_start'] = df_feat['date'].dt.is_month_start
        df_feat['days_in_month'] = df_feat['date'].dt.days_in_month

        print("Created temporal features")

    # 4. Categorical encoding
    categorical_cols = df_feat.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        # Frequency encoding
        freq_encoding = df_feat[col].value_counts().to_dict()
        df_feat[f'{col}_frequency'] = df_feat[col].map(freq_encoding)

        # Target encoding (would need target variable in practice)
        # This is just a demonstration
        if 'value' in df_feat.columns:
            target_mean = df_feat.groupby(col)['value'].mean().to_dict()
            df_feat[f'{col}_target_encoded'] = df_feat[col].map(target_mean)

    # 5. Statistical aggregations
    if len(numeric_cols) > 1:
        # Row-wise statistics
        df_feat['numeric_mean'] = df_feat[numeric_cols].mean(axis=1)
        df_feat['numeric_std'] = df_feat[numeric_cols].std(axis=1)
        df_feat['numeric_max'] = df_feat[numeric_cols].max(axis=1)
        df_feat['numeric_min'] = df_feat[numeric_cols].min(axis=1)
        df_feat['numeric_range'] = df_feat['numeric_max'] - df_feat['numeric_min']

    print(f"Original features: {df.shape[1]}")
    print(f"After engineering: {df_feat.shape[1]}")
    print(f"New features created: {df_feat.shape[1] - df.shape[1]}")

    return df_feat


# ============================================================================
# MAIN EXECUTION AND DEMONSTRATIONS
# ============================================================================

def main():
    """
    Run all demonstrations in sequence.
    """
    print("=" * 60)
    print("DATA SCIENCE INTERVIEW PREP - NUMPY & PANDAS")
    print("=" * 60)

    # NumPy demonstrations
    print("\n1. NUMPY DEMONSTRATIONS")
    print("-" * 40)

    vectorization_demo()
    broadcasting_demo()
    advanced_indexing_demo()

    # Test custom implementations
    print("=== Custom Matrix Operations ===")
    X = np.random.randn(100, 5)

    # Covariance matrix
    cov_custom = custom_covariance_matrix(X)
    cov_numpy = np.cov(X.T)
    print("Covariance matrix shape:", cov_custom.shape)
    print("Match with NumPy?", np.allclose(cov_custom, cov_numpy))

    # PCA
    X_pca, components, explained_var = manual_pca(X, n_components=2)
    print(f"\nPCA Results:")
    print(f"Transformed data shape: {X_pca.shape}")
    print(f"Explained variance: {explained_var}")

    # Pandas demonstrations
    print("\n\n2. PANDAS DEMONSTRATIONS")
    print("-" * 40)

    df = advanced_pandas_operations()
    df_clean, missing_summary = handle_missing_data(df)
    df_clean = clean_raw_data(df_clean)

    # EDA
    eda_report = generate_eda_report(df_clean)
    visualize_eda(df_clean)

    # Feature engineering
    df_features = feature_engineering_pipeline(df_clean)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Configure display options
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Run demonstrations
    main()