#!/usr/bin/env python3
"""
Data Science Interview Prep - NumPy and Pandas
==============================================
Key data manipulation concepts and common interview questions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# NUMPY ESSENTIALS
# ============================================================================

def numpy_basics():
    """Demonstrate essential NumPy operations"""
    print("=== NumPy Basics ===")
    
    # Array creation
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.zeros((3, 3))
    arr3 = np.random.randn(2, 4)
    
    print("1. Array creation:")
    print(f"   arr1: {arr1}")
    print(f"   arr2 shape: {arr2.shape}")
    print(f"   arr3:\n{arr3}")
    
    # Broadcasting
    print("\n2. Broadcasting:")
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    scalar = 2
    result = arr + scalar
    print(f"   arr + {scalar}:\n{result}")
    
    # Vectorization
    print("\n3. Vectorization:")
    arr = np.random.randn(1000000)
    
    # Loop approach (slow)
    result_loop = []
    for x in arr:
        result_loop.append(x ** 2)
    
    # Vectorized approach (fast)
    result_vec = arr ** 2
    
    print(f"   Vectorized is much faster than loops")
    print()

def numpy_advanced():
    """Demonstrate advanced NumPy operations"""
    print("=== Advanced NumPy ===")
    
    # Advanced indexing
    arr = np.arange(20).reshape(4, 5)
    print("1. Advanced indexing:")
    print(f"   Original:\n{arr}")
    
    # Boolean indexing
    mask = arr > 10
    print(f"   Elements > 10: {arr[mask]}")
    
    # Fancy indexing
    rows = [0, 2]
    cols = [1, 3]
    print(f"   Fancy indexing: {arr[rows, cols]}")
    
    # Broadcasting rules
    print("\n2. Broadcasting rules:")
    A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = np.array([10, 20, 30])  # (3,)
    print(f"   A + b:\n{A + b}")
    print()

# ============================================================================
# PANDAS ESSENTIALS
# ============================================================================

def pandas_basics():
    """Demonstrate essential Pandas operations"""
    print("=== Pandas Basics ===")
    
    # Create sample data
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, 35, 28],
        'city': ['NYC', 'LA', 'Chicago', 'Boston'],
        'salary': [50000, 60000, 70000, 55000]
    }
    df = pd.DataFrame(data)
    
    print("1. DataFrame creation:")
    print(df)
    
    # Basic operations
    print("\n2. Basic operations:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Mean age: {df['age'].mean():.1f}")
    print(f"   Max salary: {df['salary'].max()}")
    
    # Filtering
    print("\n3. Filtering:")
    high_salary = df[df['salary'] > 55000]
    print(f"   High salary employees:\n{high_salary}")
    
    # Grouping
    print("\n4. Grouping:")
    city_stats = df.groupby('city')['salary'].agg(['mean', 'count'])
    print(f"   Salary by city:\n{city_stats}")
    print()

def pandas_advanced():
    """Demonstrate advanced Pandas operations"""
    print("=== Advanced Pandas ===")
    
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(100).cumsum(),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("1. Time series operations:")
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()
    
    # Rolling window
    df['rolling_mean'] = df['value'].rolling(window=7).mean()
    
    print(f"   Rolling mean added")
    
    # Pivot tables
    print("\n2. Pivot table:")
    pivot = df.pivot_table(
        values='value', 
        index='month', 
        columns='category', 
        aggfunc='mean'
    )
    print(pivot)
    
    # Missing data handling
    print("\n3. Missing data:")
    df_with_nulls = df.copy()
    df_with_nulls.loc[10:15, 'value'] = np.nan
    
    print(f"   Nulls: {df_with_nulls['value'].isnull().sum()}")
    print(f"   Filled with mean: {df_with_nulls['value'].fillna(df_with_nulls['value'].mean()).isnull().sum()}")
    print()

# ============================================================================
# DATA CLEANING
# ============================================================================

def data_cleaning_demo():
    """Demonstrate data cleaning techniques"""
    print("=== Data Cleaning ===")
    
    # Create messy data
    messy_data = {
        'name': ['Alice', 'bob', 'Charlie', 'diana', 'Eve'],
        'age': [25, '30', 35, '28', 'unknown'],
        'salary': ['50000', '60000', '70000', '55000', '65000'],
        'date': ['2023-01-01', '2023-02-15', 'invalid', '2023-04-10', '2023-05-20']
    }
    df = pd.DataFrame(messy_data)
    
    print("1. Original messy data:")
    print(df)
    
    # Clean the data
    df_clean = df.copy()
    
    # Fix names (capitalize)
    df_clean['name'] = df_clean['name'].str.capitalize()
    
    # Fix ages (convert to numeric, handle errors)
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    
    # Fix salaries (convert to numeric)
    df_clean['salary'] = pd.to_numeric(df_clean['salary'])
    
    # Fix dates (convert to datetime, handle errors)
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    print("\n2. Cleaned data:")
    print(df_clean)
    
    # Handle missing values
    print("\n3. Missing values:")
    print(f"   Missing ages: {df_clean['age'].isnull().sum()}")
    print(f"   Missing dates: {df_clean['date'].isnull().sum()}")
    
    # Fill missing ages with median
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    print(f"   After filling: {df_clean['age'].isnull().sum()}")
    print()

# ============================================================================
# COMMON INTERVIEW QUESTIONS
# ============================================================================

def explain_vectorization():
    """Explain vectorization concept"""
    print("=== Vectorization ===")
    print("Vectorization: Apply operations to entire arrays at once")
    print("Benefits:")
    print("- Much faster than loops")
    print("- More readable code")
    print("- Leverages optimized C code")
    print("Example: arr ** 2 vs [x**2 for x in arr]")
    print()

def explain_broadcasting():
    """Explain broadcasting concept"""
    print("=== Broadcasting ===")
    print("Broadcasting: NumPy automatically handles operations between")
    print("arrays of different shapes by expanding smaller arrays")
    print("Rules:")
    print("1. Arrays must have same number of dimensions, or")
    print("2. One array must have fewer dimensions")
    print("3. Dimensions must be compatible (equal or 1)")
    print()

def explain_pandas_vs_numpy():
    """Explain when to use Pandas vs NumPy"""
    print("=== Pandas vs NumPy ===")
    print("NumPy:")
    print("- Fast numerical operations")
    print("- Multi-dimensional arrays")
    print("- Mathematical functions")
    print("- When you need raw speed")
    print()
    print("Pandas:")
    print("- Data analysis and manipulation")
    print("- Handling missing data")
    print("- Time series data")
    print("- When you need labeled data")
    print()

def demonstrate_performance():
    """Demonstrate performance differences"""
    print("=== Performance Demo ===")
    
    # Large array
    arr = np.random.randn(1000000)
    
    # NumPy vectorized
    result_numpy = arr ** 2 + 2 * arr + 1
    
    # List comprehension
    result_list = [x**2 + 2*x + 1 for x in arr]
    
    print("1. NumPy vectorized: Very fast")
    print("2. List comprehension: Much slower")
    print("3. Always prefer vectorized operations for large datasets")
    print()

def main():
    """Run all demonstrations"""
    numpy_basics()
    numpy_advanced()
    pandas_basics()
    pandas_advanced()
    data_cleaning_demo()
    explain_vectorization()
    explain_broadcasting()
    explain_pandas_vs_numpy()
    demonstrate_performance()

if __name__ == "__main__":
    main()