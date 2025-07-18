#!/usr/bin/env python3
"""
Data Science Interview Prep - Statistics
========================================
Key statistical concepts and common interview questions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def calculate_basic_stats(data):
    """Calculate basic descriptive statistics"""
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'variance': np.var(data),
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)
    }
    return stats

def detect_outliers(data):
    """Detect outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# ============================================================================
# HYPOTHESIS TESTING
# ============================================================================

def t_test(sample1, sample2, alpha=0.05):
    """Perform independent t-test"""
    # Calculate t-statistic
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    # Pooled variance
    pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
    
    # t-statistic
    t_stat = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    # p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Decision
    reject = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_null': reject,
        'alpha': alpha
    }

def chi_square_test(observed, expected=None):
    """Perform chi-square test"""
    if expected is None:
        # Assume uniform distribution
        expected = np.ones_like(observed) * np.sum(observed) / len(observed)
    
    # Chi-square statistic
    chi2_stat = np.sum((observed - expected)**2 / expected)
    
    # Degrees of freedom
    df = len(observed) - 1
    
    # p-value
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': df
    }

# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for mean"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # t-critical value
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Margin of error
    margin = t_critical * (std / np.sqrt(n))
    
    return {
        'mean': mean,
        'lower': mean - margin,
        'upper': mean + margin,
        'confidence': confidence
    }

# ============================================================================
# COMMON INTERVIEW QUESTIONS
# ============================================================================

def explain_p_value():
    """Explain p-value concept"""
    print("=== P-Value Explanation ===")
    print("P-value: Probability of observing data as extreme or more extreme")
    print("than what we observed, assuming the null hypothesis is true")
    print("Small p-value (< 0.05) suggests evidence against null hypothesis")
    print()

def explain_confidence_intervals():
    """Explain confidence intervals"""
    print("=== Confidence Intervals ===")
    print("Range of values that likely contains the true parameter")
    print("95% CI: If we repeated the study 100 times,")
    print("the true parameter would be in the interval 95 times")
    print()

def explain_type_errors():
    """Explain Type I and Type II errors"""
    print("=== Type I and Type II Errors ===")
    print("Type I Error (α): Reject null when it's true (false positive)")
    print("Type II Error (β): Fail to reject null when it's false (false negative)")
    print("Power = 1 - β: Probability of correctly rejecting false null")
    print()

def demonstrate_statistical_tests():
    """Demonstrate statistical tests"""
    print("=== Statistical Tests Demo ===")
    
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(100, 15, 30)
    group2 = np.random.normal(105, 15, 30)
    
    # Basic statistics
    print("1. Basic Statistics:")
    stats1 = calculate_basic_stats(group1)
    print(f"   Group 1 mean: {stats1['mean']:.2f}, std: {stats1['std']:.2f}")
    
    # T-test
    print("2. T-Test:")
    t_result = t_test(group1, group2)
    print(f"   t-statistic: {t_result['t_statistic']:.3f}")
    print(f"   p-value: {t_result['p_value']:.3f}")
    print(f"   Reject null: {t_result['reject_null']}")
    
    # Confidence interval
    print("3. Confidence Interval:")
    ci = confidence_interval(group1)
    print(f"   95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    
    # Chi-square test
    print("4. Chi-Square Test:")
    observed = np.array([20, 30, 25, 25])
    chi_result = chi_square_test(observed)
    print(f"   chi2-statistic: {chi_result['chi2_statistic']:.3f}")
    print(f"   p-value: {chi_result['p_value']:.3f}")
    print()

def main():
    """Run all demonstrations"""
    explain_p_value()
    explain_confidence_intervals()
    explain_type_errors()
    demonstrate_statistical_tests()

if __name__ == "__main__":
    main()