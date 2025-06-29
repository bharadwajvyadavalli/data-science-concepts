#!/usr/bin/env python3
"""
Data Science Interview Prep - Statistics
========================================
Descriptive and inferential statistics implementations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from typing import Tuple, List, Dict, Callable, Optional


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def calculate_descriptive_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate all major descriptive statistics manually.

    This demonstrates understanding of statistical concepts from first principles.
    Interview tip: Be ready to explain what each statistic represents.

    Args:
        data: 1D numpy array of numeric values

    Returns:
        Dictionary containing all calculated statistics
    """
    print("=== Calculating Descriptive Statistics from Scratch ===")

    stats = {}
    n = len(data)

    # Mean (measure of central tendency)
    stats['mean'] = np.sum(data) / n

    # Median (robust measure of central tendency)
    sorted_data = np.sort(data)
    if n % 2 == 0:
        stats['median'] = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        stats['median'] = sorted_data[n // 2]

    # Mode (most frequent value)
    unique, counts = np.unique(data, return_counts=True)
    stats['mode'] = unique[np.argmax(counts)]
    stats['mode_frequency'] = np.max(counts)

    # Variance (measure of spread)
    mean = stats['mean']
    stats['variance'] = np.sum((data - mean) ** 2) / (n - 1)  # Sample variance

    # Standard deviation (square root of variance)
    stats['std'] = np.sqrt(stats['variance'])

    # Coefficient of variation (relative variability)
    stats['cv'] = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else np.inf

    # Range
    stats['range'] = np.max(data) - np.min(data)

    # Interquartile range (IQR) - robust measure of spread
    q1 = np.percentile(sorted_data, 25)
    q3 = np.percentile(sorted_data, 75)
    stats['q1'] = q1
    stats['q3'] = q3
    stats['iqr'] = q3 - q1

    # Mean absolute deviation (MAD)
    stats['mad'] = np.mean(np.abs(data - mean))

    # Skewness (measure of asymmetry)
    # Using the standardized third moment
    stats['skewness'] = (np.sum((data - mean) ** 3) / n) / (stats['std'] ** 3)

    # Kurtosis (measure of tail heaviness)
    # Excess kurtosis (subtract 3 for normal distribution)
    stats['kurtosis'] = (np.sum((data - mean) ** 4) / n) / (stats['std'] ** 4) - 3

    # Additional percentiles
    stats['percentile_10'] = np.percentile(data, 10)
    stats['percentile_90'] = np.percentile(data, 90)

    # Print summary
    print(f"Sample size: {n}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Std Dev: {stats['std']:.4f}")
    print(f"Skewness: {stats['skewness']:.4f}")
    print(f"Kurtosis: {stats['kurtosis']:.4f}")

    return stats


def detect_outliers_multiple_methods(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Detect outliers using multiple methods.
    Important for data cleaning and understanding data quality.
    """
    print("\n=== Outlier Detection Methods ===")

    outliers = {}

    # Method 1: IQR Method (Tukey's fence)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers['iqr_method'] = data[(data < lower_fence) | (data > upper_fence)]

    # Method 2: Z-score method
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    outliers['z_score_method'] = data[z_scores > 3]  # 3 standard deviations

    # Method 3: Modified Z-score (using MAD)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    outliers['modified_z_score'] = data[np.abs(modified_z_scores) > 3.5]

    # Method 4: Isolation Forest would go here (requires sklearn)

    # Print results
    for method, outlier_values in outliers.items():
        print(f"{method}: {len(outlier_values)} outliers found")
        if len(outlier_values) > 0:
            print(f"  Values: {outlier_values[:5]}..." if len(outlier_values) > 5 else f"  Values: {outlier_values}")

    return outliers


def visualize_distribution_analysis(data: np.ndarray, title: str = "Data Distribution"):
    """
    Create comprehensive distribution visualization.
    Shows multiple aspects of the data distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Histogram with KDE
    ax = axes[0, 0]
    ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Add KDE
    kde_x = np.linspace(data.min(), data.max(), 200)
    kde = scipy_stats.gaussian_kde(data)
    ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')

    # Add vertical lines for mean and median
    ax.axvline(np.mean(data), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
    ax.axvline(np.median(data), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(data):.2f}')
    ax.set_title('Histogram with KDE')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

    # 2. Box plot
    ax = axes[0, 1]
    box_plot = ax.boxplot(data, vert=True, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    ax.set_title('Box Plot')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

    # 3. Q-Q plot (test for normality)
    ax = axes[1, 0]
    scipy_stats.probplot(data, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normal)')
    ax.grid(True, alpha=0.3)

    # 4. ECDF (Empirical Cumulative Distribution Function)
    ax = axes[1, 1]
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, ecdf, 'b-', linewidth=2, label='ECDF')

    # Add theoretical normal CDF for comparison
    x_theory = np.linspace(data.min(), data.max(), 100)
    y_theory = scipy_stats.norm.cdf(x_theory, loc=np.mean(data), scale=np.std(data))
    ax.plot(x_theory, y_theory, 'r--', linewidth=2, label='Normal CDF')
    ax.set_title('Empirical CDF')
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================================
# BOOTSTRAP AND CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_confidence_interval(
        data: np.ndarray,
        statistic_func: Callable,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        method: str = 'percentile'
) -> Tuple[float, float, float, np.ndarray]:
    """
    Calculate bootstrap confidence interval for any statistic.

    Bootstrap is a powerful resampling technique for estimating
    the sampling distribution of a statistic.

    Args:
        data: Original data
        statistic_func: Function to calculate the statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: 'percentile' or 'bca' (bias-corrected and accelerated)

    Returns:
        point_estimate: Original statistic value
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        bootstrap_distribution: All bootstrap statistics
    """
    print(f"\n=== Bootstrap Confidence Interval ({confidence_level * 100}%) ===")

    n = len(data)
    bootstrap_stats = []

    # Generate bootstrap samples
    np.random.seed(42)  # For reproducibility
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))

    bootstrap_stats = np.array(bootstrap_stats)
    point_estimate = statistic_func(data)

    if method == 'percentile':
        # Percentile method
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

    elif method == 'bca':
        # Bias-corrected and accelerated method
        # Calculate bias correction
        z0 = scipy_stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))

        # Calculate acceleration using jackknife
        jackknife_stats = []
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats.append(statistic_func(jack_sample))

        jackknife_stats = np.array(jackknife_stats)
        jack_mean = np.mean(jackknife_stats)

        # Acceleration factor
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = 6 * np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5
        acc = num / den if den != 0 else 0

        # Adjusted percentiles
        alpha = 1 - confidence_level
        z_alpha_lower = scipy_stats.norm.ppf(alpha / 2)
        z_alpha_upper = scipy_stats.norm.ppf(1 - alpha / 2)

        lower_p = scipy_stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - acc * (z0 + z_alpha_lower)))
        upper_p = scipy_stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - acc * (z0 + z_alpha_upper)))

        ci_lower = np.percentile(bootstrap_stats, lower_p * 100)
        ci_upper = np.percentile(bootstrap_stats, upper_p * 100)

    print(f"Statistic: {statistic_func.__name__}")
    print(f"Point estimate: {point_estimate:.4f}")
    print(f"CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"CI width: {ci_upper - ci_lower:.4f}")

    # Visualize bootstrap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_stats, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(point_estimate, color='red', linestyle='-', linewidth=2, label=f'Point Estimate: {point_estimate:.4f}')
    plt.axvline(ci_lower, color='green', linestyle='--', linewidth=2, label=f'CI Lower: {ci_lower:.4f}')
    plt.axvline(ci_upper, color='green', linestyle='--', linewidth=2, label=f'CI Upper: {ci_upper:.4f}')
    plt.xlabel('Statistic Value')
    plt.ylabel('Density')
    plt.title(f'Bootstrap Distribution of {statistic_func.__name__}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return point_estimate, ci_lower, ci_upper, bootstrap_stats


# ============================================================================
# INFERENTIAL STATISTICS
# ============================================================================

def t_test_from_scratch(
        sample1: np.ndarray,
        sample2: np.ndarray,
        paired: bool = False,
        alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Implement t-test from scratch (both independent and paired).

    T-test compares means between two groups.
    Assumptions: Normal distribution, equal variances (for independent)

    Args:
        sample1, sample2: Data arrays
        paired: If True, performs paired t-test
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with test statistics and p-value
    """
    print("\n=== T-Test Implementation ===")
    print(f"Test type: {'Paired' if paired else 'Independent'} samples")

    results = {}

    if paired:
        # Paired t-test
        if len(sample1) != len(sample2):
            raise ValueError("Paired samples must have equal length")

        differences = sample1 - sample2
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_diff = std_diff / np.sqrt(n)

        t_stat = mean_diff / se_diff
        df = n - 1

        results['mean_difference'] = mean_diff
        results['std_difference'] = std_diff

    else:
        # Independent samples t-test
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

        # Pooled standard error (assumes equal variances)
        pooled_se = np.sqrt(var1 / n1 + var2 / n2)
        t_stat = (mean1 - mean2) / pooled_se

        # Degrees of freedom (Welch's approximation for unequal variances)
        df = (var1 / n1 + var2 / n2) ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))

        results['mean1'] = mean1
        results['mean2'] = mean2
        results['mean_difference'] = mean1 - mean2
        results['pooled_se'] = pooled_se

    # Calculate p-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))
    elif alternative == 'less':
        p_value = scipy_stats.t.cdf(t_stat, df)
    else:  # greater
        p_value = 1 - scipy_stats.t.cdf(t_stat, df)

    # Effect size (Cohen's d)
    if paired:
        cohens_d = mean_diff / std_diff
    else:
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std

    results['t_statistic'] = t_stat
    results['degrees_of_freedom'] = df
    results['p_value'] = p_value
    results['cohens_d'] = cohens_d

    # Print results
    print(f"t-statistic: {t_stat:.4f}")
    print(f"Degrees of freedom: {df:.2f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

    return results


def anova_from_scratch(*groups) -> Dict[str, float]:
    """
    One-way ANOVA (Analysis of Variance) from scratch.

    ANOVA tests if means differ across multiple groups.
    H0: All group means are equal
    H1: At least one group mean differs

    Returns:
        Dictionary with F-statistic, p-value, and other statistics
    """
    print("\n=== One-Way ANOVA Implementation ===")

    # Validate input
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    # Convert to numpy arrays
    groups = [np.array(group) for group in groups]

    # Calculate overall statistics
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    n_total = len(all_data)
    k = len(groups)  # number of groups

    # Calculate sum of squares
    # Total sum of squares (SST)
    ss_total = np.sum((all_data - grand_mean) ** 2)

    # Between-group sum of squares (SSB)
    ss_between = 0
    group_means = []
    group_sizes = []

    for group in groups:
        group_mean = np.mean(group)
        group_size = len(group)
        group_means.append(group_mean)
        group_sizes.append(group_size)
        ss_between += group_size * (group_mean - grand_mean) ** 2

    # Within-group sum of squares (SSW)
    ss_within = ss_total - ss_between

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1

    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # F-statistic
    f_stat = ms_between / ms_within

    # P-value
    p_value = 1 - scipy_stats.f.cdf(f_stat, df_between, df_within)

    # Effect size (eta-squared)
    eta_squared = ss_between / ss_total

    # Omega-squared (less biased effect size)
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)

    results = {
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_between': df_between,
        'df_within': df_within,
        'ss_between': ss_between,
        'ss_within': ss_within,
        'ss_total': ss_total,
        'ms_between': ms_between,
        'ms_within': ms_within,
        'eta_squared': eta_squared,
        'omega_squared': omega_squared,
        'group_means': group_means,
        'group_sizes': group_sizes,
        'grand_mean': grand_mean
    }

    # Print ANOVA table
    print("\nANOVA Table:")
    print("-" * 60)
    print(f"{'Source':<15} {'SS':<12} {'df':<8} {'MS':<12} {'F':<10} {'p-value':<10}")
    print("-" * 60)
    print(
        f"{'Between Groups':<15} {ss_between:<12.4f} {df_between:<8} {ms_between:<12.4f} {f_stat:<10.4f} {p_value:<10.4f}")
    print(f"{'Within Groups':<15} {ss_within:<12.4f} {df_within:<8} {ms_within:<12.4f}")
    print(f"{'Total':<15} {ss_total:<12.4f} {df_total:<8}")
    print("-" * 60)
    print(f"\nEffect size (η²): {eta_squared:.4f}")
    print(f"Effect size (ω²): {omega_squared:.4f}")
    print(f"Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

    return results


def chi_squared_test(observed: np.ndarray, expected: np.ndarray = None) -> Dict[str, float]:
    """
    Chi-squared test implementation.

    Can be used for:
    1. Goodness of fit (1D)
    2. Test of independence (2D contingency table)

    Args:
        observed: Observed frequencies (1D or 2D array)
        expected: Expected frequencies (if None, calculated for independence)

    Returns:
        Dictionary with test statistics
    """
    print("\n=== Chi-Squared Test Implementation ===")

    observed = np.array(observed)

    if observed.ndim == 1:
        # Goodness of fit test
        print("Test type: Goodness of fit")

        if expected is None:
            # Assume uniform distribution
            n_categories = len(observed)
            total = np.sum(observed)
            expected = np.full(n_categories, total / n_categories)
        else:
            expected = np.array(expected)

        # Chi-squared statistic
        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1

    else:
        # Test of independence (contingency table)
        print("Test type: Independence test")

        # Calculate expected frequencies
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        total = observed.sum()

        expected = np.outer(row_totals, col_totals) / total

        # Chi-squared statistic
        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = (observed.shape[0] - 1) * (observed.shape[1] - 1)

        # Cramér's V (effect size for contingency tables)
        n = total
        min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim))

    # P-value
    p_value = 1 - scipy_stats.chi2.cdf(chi2, df)

    # Critical value at α = 0.05
    critical_value = scipy_stats.chi2.ppf(0.95, df)

    results = {
        'chi2_statistic': chi2,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'critical_value': critical_value
    }

    if observed.ndim == 2:
        results['cramers_v'] = cramers_v

    # Print results
    print(f"Chi-squared statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {df}")
    print(f"p-value: {p_value:.4f}")
    print(f"Critical value (α=0.05): {critical_value:.4f}")
    if observed.ndim == 2:
        print(f"Cramér's V: {cramers_v:.4f}")
    print(f"Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

    return results


# ============================================================================
# PROBABILITY DISTRIBUTIONS
# ============================================================================

def demonstrate_distributions():
    """
    Demonstrate working with various probability distributions.
    Shows PDF, CDF, and sampling.
    """
    print("\n=== Probability Distributions Demo ===")

    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 1. Normal Distribution
    ax = axes[0]
    x = np.linspace(-4, 4, 1000)

    # Different parameters
    for mu, sigma in [(0, 1), (0, 0.5), (0, 2), (2, 1)]:
        y = scipy_stats.norm.pdf(x, loc=mu, scale=sigma)
        ax.plot(x, y, label=f'μ={mu}, σ={sigma}')

    ax.set_title('Normal Distribution PDF')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Binomial Distribution
    ax = axes[1]
    n, p = 20, 0.3
    x = np.arange(0, n + 1)
    y = scipy_stats.binom.pmf(x, n, p)
    ax.bar(x, y, alpha=0.8)
    ax.set_title(f'Binomial Distribution PMF (n={n}, p={p})')
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Probability')
    ax.grid(True, alpha=0.3)

    # 3. Poisson Distribution
    ax = axes[2]
    lambdas = [1, 3, 5]
    x = np.arange(0, 15)

    for lam in lambdas:
        y = scipy_stats.poisson.pmf(x, lam)
        ax.plot(x, y, 'o-', label=f'λ={lam}')

    ax.set_title('Poisson Distribution PMF')
    ax.set_xlabel('k')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Exponential Distribution
    ax = axes[3]
    x = np.linspace(0, 5, 1000)

    for lam in [0.5, 1, 2]:
        y = scipy_stats.expon.pdf(x, scale=1 / lam)
        ax.plot(x, y, label=f'λ={lam}')

    ax.set_title('Exponential Distribution PDF')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Chi-squared Distribution
    ax = axes[4]
    x = np.linspace(0, 20, 1000)

    for df in [1, 3, 5, 10]:
        y = scipy_stats.chi2.pdf(x, df)
        ax.plot(x, y, label=f'df={df}')

    ax.set_title('Chi-squared Distribution PDF')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Beta Distribution
    ax = axes[5]
    x = np.linspace(0, 1, 1000)

    params = [(0.5, 0.5), (2, 2), (2, 5), (5, 2)]
    for a, b in params:
        y = scipy_stats.beta.pdf(x, a, b)
        ax.plot(x, y, label=f'α={a}, β={b}')

    ax.set_title('Beta Distribution PDF')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Common Probability Distributions', fontsize=16)
    plt.tight_layout()
    plt.show()


def test_distribution_fit(data: np.ndarray, dist_name: str = 'norm') -> Dict[str, any]:
    """
    Test if data follows a specific distribution using multiple tests.

    Args:
        data: Sample data
        dist_name: Distribution name ('norm', 'expon', 'gamma', etc.)

    Returns:
        Dictionary with test results
    """
    print(f"\n=== Testing Fit to {dist_name} Distribution ===")

    results = {}

    # 1. Kolmogorov-Smirnov test
    if dist_name == 'norm':
        # Estimate parameters
        params = scipy_stats.norm.fit(data)
        ks_stat, ks_pvalue = scipy_stats.kstest(data, 'norm', args=params)
    elif dist_name == 'expon':
        params = scipy_stats.expon.fit(data)
        ks_stat, ks_pvalue = scipy_stats.kstest(data, 'expon', args=params)
    elif dist_name == 'gamma':
        params = scipy_stats.gamma.fit(data)
        ks_stat, ks_pvalue = scipy_stats.kstest(data, 'gamma', args=params)

    results['ks_statistic'] = ks_stat
    results['ks_pvalue'] = ks_pvalue
    results['parameters'] = params

    # 2. Anderson-Darling test (more sensitive in tails)
    if dist_name in ['norm', 'expon', 'gumbel']:
        ad_result = scipy_stats.anderson(data, dist=dist_name)
        results['ad_statistic'] = ad_result.statistic
        results['ad_critical_values'] = ad_result.critical_values
        results['ad_significance_levels'] = ad_result.significance_level

    # 3. Shapiro-Wilk test (specifically for normality)
    if dist_name == 'norm' and len(data) <= 5000:
        sw_stat, sw_pvalue = scipy_stats.shapiro(data)
        results['shapiro_statistic'] = sw_stat
        results['shapiro_pvalue'] = sw_pvalue

    # 4. Visual assessment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with fitted distribution
    ax1.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Plot fitted distribution
    x_range = np.linspace(data.min(), data.max(), 100)
    if dist_name == 'norm':
        fitted_pdf = scipy_stats.norm.pdf(x_range, *params)
    elif dist_name == 'expon':
        fitted_pdf = scipy_stats.expon.pdf(x_range, *params)
    elif dist_name == 'gamma':
        fitted_pdf = scipy_stats.gamma.pdf(x_range, *params)

    ax1.plot(x_range, fitted_pdf, 'r-', linewidth=2, label=f'Fitted {dist_name}')
    ax1.set_title('Histogram with Fitted Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()

    # Q-Q plot
    if dist_name == 'norm':
        scipy_stats.probplot(data, dist="norm", plot=ax2)
    elif dist_name == 'expon':
        scipy_stats.probplot(data, dist="expon", plot=ax2)

    ax2.set_title(f'Q-Q Plot ({dist_name})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Kolmogorov-Smirnov test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_pvalue:.4f}")
    print(f"  Reject H0 (data follows {dist_name})? {'Yes' if ks_pvalue < 0.05 else 'No'}")

    if 'shapiro_pvalue' in results:
        print(f"\nShapiro-Wilk test (normality):")
        print(f"  Statistic: {results['shapiro_statistic']:.4f}")
        print(f"  p-value: {results['shapiro_pvalue']:.4f}")

    return results


# ============================================================================
# CENTRAL LIMIT THEOREM
# ============================================================================

def simulate_clt(
        population_dist: str = 'exponential',
        sample_sizes: List[int] = [1, 5, 10, 30, 100],
        n_samples: int = 10000
):
    """
    Simulate Central Limit Theorem with various distributions.

    CLT states that the sampling distribution of the mean approaches
    normal distribution as sample size increases, regardless of the
    population distribution.
    """
    print("\n=== Central Limit Theorem Simulation ===")
    print(f"Population distribution: {population_dist}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Number of samples: {n_samples}")

    # Generate population based on distribution type
    if population_dist == 'exponential':
        population = scipy_stats.expon.rvs(scale=2, size=100000)
    elif population_dist == 'uniform':
        population = scipy_stats.uniform.rvs(loc=0, scale=10, size=100000)
    elif population_dist == 'bimodal':
        # Create bimodal distribution
        pop1 = scipy_stats.norm.rvs(loc=2, scale=1, size=50000)
        pop2 = scipy_stats.norm.rvs(loc=8, scale=1, size=50000)
        population = np.concatenate([pop1, pop2])

    # True population parameters
    pop_mean = np.mean(population)
    pop_std = np.std(population)

    # Set up plotting
    n_plots = len(sample_sizes) + 1
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot population distribution
    ax = axes[0]
    ax.hist(population[:10000], bins=50, density=True, alpha=0.7, color='gray')
    ax.set_title(f'Population Distribution ({population_dist})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.axvline(pop_mean, color='red', linestyle='--', label=f'μ={pop_mean:.2f}')
    ax.legend()

    # Simulate for different sample sizes
    for idx, sample_size in enumerate(sample_sizes):
        sample_means = []

        # Generate many samples and calculate means
        for _ in range(n_samples):
            sample = np.random.choice(population, size=sample_size, replace=True)
            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)

        # Plot sampling distribution
        ax = axes[idx + 1]
        ax.hist(sample_means, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

        # Theoretical normal distribution
        theoretical_mean = pop_mean
        theoretical_std = pop_std / np.sqrt(sample_size)
        x_range = np.linspace(sample_means.min(), sample_means.max(), 100)
        theoretical_pdf = scipy_stats.norm.pdf(x_range, loc=theoretical_mean, scale=theoretical_std)
        ax.plot(x_range, theoretical_pdf, 'r-', linewidth=2, label='Theoretical Normal')

        ax.set_title(f'Sampling Distribution (n={sample_size})')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.legend()

        # Calculate normality test
        if len(sample_means) <= 5000:
            _, p_value = scipy_stats.shapiro(sample_means)
            ax.text(0.05, 0.95, f'Shapiro p={p_value:.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Central Limit Theorem Demonstration', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Demonstrate convergence
    print("\nConvergence to normality (Shapiro-Wilk p-values):")
    for sample_size in sample_sizes:
        sample_means = [np.mean(np.random.choice(population, size=sample_size, replace=True))
                        for _ in range(1000)]
        if len(sample_means) <= 5000:
            _, p_value = scipy_stats.shapiro(sample_means)
            print(f"  n={sample_size}: p={p_value:.4f}")


# ============================================================================
# POWER ANALYSIS
# ============================================================================

def power_analysis_simulation(
        effect_size: float = 0.5,
        alpha: float = 0.05,
        sample_sizes: List[int] = [10, 20, 30, 50, 100],
        n_simulations: int = 1000
) -> Dict[int, float]:
    """
    Simulate statistical power for different sample sizes.

    Power = probability of correctly rejecting false null hypothesis

    Args:
        effect_size: Cohen's d (standardized effect size)
        alpha: Significance level
        sample_sizes: List of sample sizes to test
        n_simulations: Number of simulations per sample size

    Returns:
        Dictionary mapping sample size to power
    """
    print("\n=== Statistical Power Analysis ===")
    print(f"Effect size (Cohen's d): {effect_size}")
    print(f"Significance level (α): {alpha}")

    power_results = {}

    for n in sample_sizes:
        significant_results = 0

        for _ in range(n_simulations):
            # Generate data under alternative hypothesis
            # Group 1: mean = 0
            group1 = np.random.normal(0, 1, n)
            # Group 2: mean = effect_size (in standard deviation units)
            group2 = np.random.normal(effect_size, 1, n)

            # Perform t-test
            _, p_value = scipy_stats.ttest_ind(group1, group2)

            if p_value < alpha:
                significant_results += 1

        power = significant_results / n_simulations
        power_results[n] = power

        print(f"n={n}: Power = {power:.3f}")

    # Visualize power curve
    plt.figure(figsize=(10, 6))
    sample_sizes_array = np.array(list(power_results.keys()))
    powers = np.array(list(power_results.values()))

    plt.plot(sample_sizes_array, powers, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='r', linestyle='--', label='Power = 0.8 (conventional threshold)')
    plt.xlabel('Sample Size per Group')
    plt.ylabel('Statistical Power')
    plt.title(f'Power Analysis (Effect Size = {effect_size}, α = {alpha})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)

    # Find sample size for 80% power
    if any(p >= 0.8 for p in powers):
        n_for_80_power = min(n for n, p in power_results.items() if p >= 0.8)
        plt.axvline(x=n_for_80_power, color='g', linestyle=':',
                    label=f'n={n_for_80_power} for 80% power')
        plt.legend()

    plt.show()

    return power_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run all statistical demonstrations.
    """
    print("=" * 60)
    print("DATA SCIENCE INTERVIEW PREP - STATISTICS")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)

    # 1. Descriptive Statistics
    print("\n1. DESCRIPTIVE STATISTICS")
    print("-" * 40)

    # Generate different types of data
    normal_data = np.random.normal(100, 15, 1000)
    skewed_data = np.random.gamma(2, 2, 1000)
    bimodal_data = np.concatenate([np.random.normal(50, 10, 500),
                                   np.random.normal(100, 10, 500)])

    # Calculate statistics
    stats_normal = calculate_descriptive_stats(normal_data)
    visualize_distribution_analysis(normal_data, "Normal Distribution (μ=100, σ=15)")

    # Outlier detection
    outliers = detect_outliers_multiple_methods(normal_data)

    # 2. Bootstrap and Confidence Intervals
    print("\n\n2. BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 40)

    # Bootstrap for different statistics
    _, _, _, _ = bootstrap_confidence_interval(normal_data, np.mean, n_bootstrap=5000)
    _, _, _, _ = bootstrap_confidence_interval(normal_data, np.median, n_bootstrap=5000)

    # Custom statistic (trimmed mean)
    def trimmed_mean(x, trim_pct=0.1):
        trimmed = scipy_stats.trim_mean(x, trim_pct)
        return trimmed

    _, _, _, _ = bootstrap_confidence_interval(normal_data, trimmed_mean, n_bootstrap=5000)

    # 3. Inferential Statistics
    print("\n\n3. INFERENTIAL STATISTICS")
    print("-" * 40)

    # T-test examples
    group1 = np.random.normal(100, 15, 50)
    group2 = np.random.normal(105, 15, 50)  # Small effect
    t_results = t_test_from_scratch(group1, group2)

    # Paired t-test
    before = np.random.normal(100, 10, 30)
    after = before + np.random.normal(5, 5, 30)  # Treatment effect
    paired_results = t_test_from_scratch(before, after, paired=True)

    # ANOVA
    groups_anova = [
        np.random.normal(100, 10, 30),
        np.random.normal(105, 10, 30),
        np.random.normal(95, 10, 30),
        np.random.normal(102, 10, 30)
    ]
    anova_results = anova_from_scratch(*groups_anova)

    # Chi-squared test
    # Test of independence example
    observed_table = np.array([
        [20, 30, 25],
        [30, 40, 35],
        [25, 35, 40]
    ])
    chi2_results = chi_squared_test(observed_table)

    # 4. Probability Distributions
    print("\n\n4. PROBABILITY DISTRIBUTIONS")
    print("-" * 40)

    demonstrate_distributions()

    # Test distribution fit
    exponential_data = np.random.exponential(scale=2, size=1000)
    test_distribution_fit(exponential_data, 'expon')
    test_distribution_fit(normal_data, 'norm')

    # 5. Central Limit Theorem
    print("\n\n5. CENTRAL LIMIT THEOREM")
    print("-" * 40)

    simulate_clt('exponential')
    simulate_clt('uniform')

    # 6. Power Analysis
    print("\n\n6. STATISTICAL POWER")
    print("-" * 40)

    # Small, medium, large effect sizes
    for effect_size in [0.2, 0.5, 0.8]:
        print(
            f"\nEffect size: {effect_size} ({'small' if effect_size == 0.2 else 'medium' if effect_size == 0.5 else 'large'})")
        power_analysis_simulation(effect_size=effect_size, n_simulations=1000)

    print("\n" + "=" * 60)
    print("STATISTICS DEMONSTRATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Run all demonstrations
    main()