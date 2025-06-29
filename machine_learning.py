#!/usr/bin/env python3
"""
Data Science Interview Prep - Machine Learning
=============================================
Statistical learning and ML algorithm implementations from scratch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression, make_blobs
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# LINEAR REGRESSION FROM SCRATCH
# ============================================================================

class LinearRegressionScratch:
    """
    Linear Regression implementation using gradient descent.

    Interview tips:
    - Explain cost function (MSE)
    - Discuss gradient descent vs normal equation
    - Mention regularization (Ridge/Lasso)
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: Optional[str] = None, lambda_reg: float = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization  # None, 'l2' (Ridge), 'l1' (Lasso)
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _add_regularization_gradient(self, dw: np.ndarray) -> np.ndarray:
        """Add regularization term to weight gradients."""
        if self.regularization == 'l2':
            # Ridge regression
            dw += self.lambda_reg * self.weights
        elif self.regularization == 'l1':
            # Lasso regression
            dw += self.lambda_reg * np.sign(self.weights)
        return dw

    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate loss with optional regularization."""
        mse = np.mean((y_true - y_pred) ** 2)

        if self.regularization == 'l2':
            reg_term = self.lambda_reg * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            reg_term = self.lambda_reg * np.sum(np.abs(self.weights))
        else:
            reg_term = 0

        return mse + reg_term

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Fit linear regression using gradient descent.

        Cost function: J(θ) = (1/2m) Σ(h(x) - y)² + regularization
        Gradient: ∂J/∂θ = (1/m) X^T(Xθ - y) + regularization
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Calculate loss
            loss = self._calculate_loss(y, y_pred)
            self.loss_history.append(loss)

            # Calculate gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Add regularization to weight gradient
            dw = self._add_regularization_gradient(dw)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return X @ self.weights + self.bias

    def normal_equation(self, X: np.ndarray, y: np.ndarray):
        """
        Solve using normal equation (closed-form solution).
        θ = (X^T X)^(-1) X^T y

        For Ridge: θ = (X^T X + λI)^(-1) X^T y
        """
        # Add bias term to X
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]

        if self.regularization == 'l2':
            # Ridge regression closed form
            lambda_identity = self.lambda_reg * np.eye(X_with_bias.shape[1])
            lambda_identity[0, 0] = 0  # Don't regularize bias
            theta = np.linalg.inv(X_with_bias.T @ X_with_bias + lambda_identity) @ X_with_bias.T @ y
        else:
            # Standard linear regression
            theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def plot_convergence(self):
        """Plot loss convergence during training."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.title('Training Loss Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()


def compare_linear_regression_methods():
    """
    Compare different linear regression implementations.
    Important for interviews to show understanding of trade-offs.
    """
    print("=== Comparing Linear Regression Methods ===")

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_weights = np.array([2, -1, 0.5])
    y = X @ true_weights + 3 + np.random.randn(100) * 0.1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Gradient Descent
    print("\n1. Gradient Descent Implementation:")
    lr_gd = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
    lr_gd.fit(X_train, y_train, verbose=False)
    y_pred_gd = lr_gd.predict(X_test)
    r2_gd = lr_gd.r2_score(y_test, y_pred_gd)
    print(f"R² score: {r2_gd:.4f}")
    print(f"Weights: {lr_gd.weights}")

    # 2. Normal Equation
    print("\n2. Normal Equation Implementation:")
    lr_ne = LinearRegressionScratch()
    lr_ne.normal_equation(X_train, y_train)
    y_pred_ne = lr_ne.predict(X_test)
    r2_ne = lr_ne.r2_score(y_test, y_pred_ne)
    print(f"R² score: {r2_ne:.4f}")
    print(f"Weights: {lr_ne.weights}")

    # 3. Ridge Regression
    print("\n3. Ridge Regression (L2):")
    lr_ridge = LinearRegressionScratch(regularization='l2', lambda_reg=0.1)
    lr_ridge.fit(X_train, y_train, verbose=False)
    y_pred_ridge = lr_ridge.predict(X_test)
    r2_ridge = lr_ridge.r2_score(y_test, y_pred_ridge)
    print(f"R² score: {r2_ridge:.4f}")
    print(f"Weights: {lr_ridge.weights}")

    # 4. Compare with sklearn
    from sklearn.linear_model import LinearRegression
    lr_sklearn = LinearRegression()
    lr_sklearn.fit(X_train, y_train)
    r2_sklearn = lr_sklearn.score(X_test, y_test)
    print(f"\n4. Sklearn R² score: {r2_sklearn:.4f}")
    print(f"Sklearn weights: {lr_sklearn.coef_}")

    # Visualize predictions
    plt.figure(figsize=(12, 8))

    methods = ['Gradient Descent', 'Normal Equation', 'Ridge', 'Sklearn']
    predictions = [y_pred_gd, y_pred_ne, y_pred_ridge, lr_sklearn.predict(X_test)]

    for i, (method, pred) in enumerate(zip(methods, predictions)):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_test, pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{method}')
        plt.grid(True, alpha=0.3)

        # Add R² score
        r2 = lr_gd.r2_score(y_test, pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.show()


# ============================================================================
# LOGISTIC REGRESSION FROM SCRATCH
# ============================================================================

class LogisticRegressionScratch:
    """
    Logistic Regression implementation from scratch.

    Key concepts:
    - Sigmoid activation
    - Log loss (cross-entropy)
    - Gradient descent for classification
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: Optional[str] = None, lambda_reg: float = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        σ(z) = 1 / (1 + e^(-z))
        """
        # Clip to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary cross-entropy loss with optional regularization.
        L = -1/m Σ[y*log(ŷ) + (1-y)*log(1-ŷ)] + regularization
        """
        epsilon = 1e-7  # Small value to prevent log(0)
        bce = -np.mean(y_true * np.log(y_pred + epsilon) +
                       (1 - y_true) * np.log(1 - y_pred + epsilon))

        if self.regularization == 'l2':
            reg_term = (self.lambda_reg / (2 * len(y_true))) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            reg_term = (self.lambda_reg / len(y_true)) * np.sum(np.abs(self.weights))
        else:
            reg_term = 0

        return bce + reg_term

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Fit logistic regression using gradient descent."""
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # Calculate loss
            loss = self._calculate_loss(y, y_pred)
            self.loss_history.append(loss)

            # Calculate gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Add regularization
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def decision_boundary(self, X: np.ndarray, y: np.ndarray,
                          feature_names: Optional[List[str]] = None):
        """Visualize decision boundary for 2D data."""
        if X.shape[1] != 2:
            print("Decision boundary visualization requires 2D data")
            return

        plt.figure(figsize=(10, 8))

        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Predict on mesh
        Z = self.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                              edgecolors='black', s=100)

        plt.xlabel(feature_names[0] if feature_names else 'Feature 1')
        plt.ylabel(feature_names[1] if feature_names else 'Feature 2')
        plt.title('Logistic Regression Decision Boundary')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.show()


# ============================================================================
# BIAS-VARIANCE TRADEOFF
# ============================================================================

def visualize_bias_variance_tradeoff():
    """
    Visualize bias-variance tradeoff with polynomial regression.
    Critical concept for ML interviews.
    """
    print("\n=== Bias-Variance Tradeoff Visualization ===")

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import validation_curve

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.sort(np.random.rand(n_samples) * 2 - 1).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + np.random.randn(n_samples) * 0.1

    # Test different polynomial degrees
    degrees = range(1, 15)

    # Calculate training and validation errors
    train_scores, val_scores = validation_curve(
        Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear', LinearRegression())
        ]),
        X, y,
        param_name='poly__degree',
        param_range=degrees,
        cv=5,
        scoring='neg_mean_squared_error'
    )

    train_errors = -train_scores.mean(axis=1)
    val_errors = -val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    # Plot bias-variance tradeoff
    plt.figure(figsize=(12, 5))

    # Left plot: Training and validation curves
    plt.subplot(1, 2, 1)
    plt.plot(degrees, train_errors, 'b-', label='Training error', linewidth=2)
    plt.fill_between(degrees, train_errors - train_std, train_errors + train_std,
                     alpha=0.1, color='blue')
    plt.plot(degrees, val_errors, 'r-', label='Validation error', linewidth=2)
    plt.fill_between(degrees, val_errors - val_std, val_errors + val_std,
                     alpha=0.1, color='red')

    # Find optimal degree
    optimal_degree = degrees[np.argmin(val_errors)]
    plt.axvline(x=optimal_degree, color='g', linestyle='--',
                label=f'Optimal degree = {optimal_degree}')

    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Complexity vs Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Add annotations for regions
    plt.text(2, np.max(val_errors) * 0.8, 'High Bias\n(Underfitting)',
             fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='yellow'))
    plt.text(12, np.max(val_errors) * 0.8, 'High Variance\n(Overfitting)',
             fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='orange'))

    # Right plot: Show fitted curves for different complexities
    plt.subplot(1, 2, 2)
    X_plot = np.linspace(-1, 1, 300).reshape(-1, 1)

    for degree in [1, optimal_degree, 13]:
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        poly_model.fit(X, y)
        y_plot = poly_model.predict(X_plot)

        label = f'Degree {degree}'
        if degree == 1:
            label += ' (Underfit)'
        elif degree == optimal_degree:
            label += ' (Optimal)'
        else:
            label += ' (Overfit)'

        plt.plot(X_plot, y_plot, linewidth=2, label=label)

    plt.scatter(X, y, alpha=0.6, s=40, c='black', label='Training data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Additional insights
    print(f"Optimal polynomial degree: {optimal_degree}")
    print(f"Training error at optimal degree: {train_errors[optimal_degree - 1]:.4f}")
    print(f"Validation error at optimal degree: {val_errors[optimal_degree - 1]:.4f}")


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                     y_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate comprehensive classification metrics.
    Essential for evaluating model performance in interviews.
    """
    from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score, roc_curve,
                                 precision_recall_curve, classification_report)

    print("\n=== Classification Metrics ===")

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['true_negatives'] = cm[0, 0]
    metrics['false_positives'] = cm[0, 1]
    metrics['false_negatives'] = cm[1, 0]
    metrics['true_positives'] = cm[1, 1]

    # Additional metrics
    metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2

    # Matthews Correlation Coefficient
    mcc_num = (cm[1, 1] * cm[0, 0]) - (cm[0, 1] * cm[1, 0])
    mcc_den = np.sqrt((cm[1, 1] + cm[0, 1]) * (cm[1, 1] + cm[1, 0]) *
                      (cm[0, 0] + cm[0, 1]) * (cm[0, 0] + cm[1, 0]))
    metrics['mcc'] = mcc_num / mcc_den if mcc_den != 0 else 0

    # If probabilities are provided
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        metrics['pr_curve'] = {'precision': precision, 'recall': recall,
                               'thresholds': pr_thresholds}

        # Average precision (area under PR curve)
        from sklearn.metrics import average_precision_score
        metrics['average_precision'] = average_precision_score(y_true, y_proba)

    # Print summary
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    return metrics


def plot_classification_metrics(metrics: Dict):
    """Comprehensive visualization of classification metrics."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')

    # 2. ROC Curve
    if 'roc_curve' in metrics:
        ax2 = plt.subplot(2, 3, 2)
        roc = metrics['roc_curve']
        plt.plot(roc['fpr'], roc['tpr'], 'b-', linewidth=2,
                 label=f"AUC = {metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    if 'pr_curve' in metrics:
        ax3 = plt.subplot(2, 3, 3)
        pr = metrics['pr_curve']
        plt.plot(pr['recall'], pr['precision'], 'g-', linewidth=2,
                 label=f"AP = {metrics['average_precision']:.3f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 4. Metrics Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_values = [metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1_score'], metrics['specificity']]

    bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    plt.title('Classification Metrics')

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # 5. Threshold Analysis (if probabilities available)
    if 'roc_curve' in metrics:
        ax5 = plt.subplot(2, 3, 5)
        thresholds = metrics['roc_curve']['thresholds'][::10]  # Sample thresholds
        tpr = metrics['roc_curve']['tpr'][::10]
        fpr = metrics['roc_curve']['fpr'][::10]

        plt.plot(thresholds, tpr, 'b-', label='True Positive Rate')
        plt.plot(thresholds, 1 - fpr, 'r-', label='True Negative Rate')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Threshold vs TPR/TNR')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 6. Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    Classification Report:

    Total Samples: {sum(sum(row) for row in metrics['confusion_matrix'])}

    True Positives:  {metrics['true_positives']}
    True Negatives:  {metrics['true_negatives']}
    False Positives: {metrics['false_positives']}
    False Negatives: {metrics['false_negatives']}

    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}
    MCC: {metrics['mcc']:.4f}
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center')

    plt.suptitle('Classification Performance Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================================
# DECISION TREE FROM SCRATCH
# ============================================================================

class DecisionTreeNode:
    """Node class for decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, samples=None, impurity=None):
        self.feature = feature  # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class value (for leaf nodes)
        self.samples = samples  # Number of samples at this node
        self.impurity = impurity  # Impurity at this node


class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier implementation from scratch.

    Key concepts:
    - Information gain / Gini impurity
    - Recursive tree building
    - Pruning strategies
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 criterion: str = 'gini', min_impurity_decrease: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion  # 'gini' or 'entropy'
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.n_features_ = None
        self.n_classes_ = None
        self.feature_importances_ = None

    def gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity.
        Gini = 1 - Σ(p_i)²
        """
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy.
        Entropy = -Σ(p_i * log(p_i))
        """
        proportions = np.bincount(y) / len(y)
        # Remove zero proportions to avoid log(0)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))

    def calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on chosen criterion."""
        if self.criterion == 'gini':
            return self.gini_impurity(y)
        else:
            return self.entropy(y)

    def information_gain(self, parent: np.ndarray, left: np.ndarray,
                         right: np.ndarray) -> float:
        """
        Calculate information gain from split.
        IG = impurity(parent) - weighted_avg(impurity(children))
        """
        n = len(parent)
        n_left, n_right = len(left), len(right)

        if n_left == 0 or n_right == 0:
            return 0

        parent_impurity = self.calculate_impurity(parent)
        left_impurity = self.calculate_impurity(left)
        right_impurity = self.calculate_impurity(right)

        weighted_impurity = (n_left / n * left_impurity +
                             n_right / n * right_impurity)

        return parent_impurity - weighted_impurity

    def best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best split for a node.
        Returns: (best_feature, best_threshold, best_gain)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        current_impurity = self.calculate_impurity(y)

        # Try all features
        for feature in range(self.n_features_):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature])

            # Try all possible thresholds
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_split or \
                        np.sum(right_mask) < self.min_samples_split:
                    continue

                # Calculate information gain
                gain = self.information_gain(y, y[left_mask], y[right_mask])

                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Calculate current node impurity
        node_impurity = self.calculate_impurity(y)

        # Stopping criteria
        if (depth >= self.max_depth or
                n_classes == 1 or
                n_samples < self.min_samples_split):
            # Create leaf node
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(
                value=leaf_value,
                samples=n_samples,
                impurity=node_impurity
            )

        # Find best split
        best_feature, best_threshold, best_gain = self.best_split(X, y)

        # Check if gain is sufficient
        if best_feature is None or best_gain < self.min_impurity_decrease:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(
                value=leaf_value,
                samples=n_samples,
                impurity=node_impurity
            )

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        # Update feature importance
        self.feature_importances_[best_feature] += (
                n_samples * node_impurity -
                left_subtree.samples * left_subtree.impurity -
                right_subtree.samples * right_subtree.impurity
        )

        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            samples=n_samples,
            impurity=node_impurity
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the decision tree."""
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.feature_importances_ = np.zeros(self.n_features_)

        self.root = self.build_tree(X, y)

        # Normalize feature importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()

    def predict_sample(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Predict a single sample by traversing the tree."""
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict multiple samples."""
        return np.array([self.predict_sample(x, self.root) for x in X])

    def print_tree(self, node: DecisionTreeNode = None, depth: int = 0):
        """Print the tree structure."""
        if node is None:
            node = self.root

        indent = "  " * depth

        if node.value is not None:
            print(f"{indent}Leaf: Class {node.value} (samples: {node.samples})")
        else:
            print(f"{indent}Node: Feature {node.feature} <= {node.threshold:.4f}")
            print(f"{indent}  (samples: {node.samples}, impurity: {node.impurity:.4f})")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    def get_depth(self, node: DecisionTreeNode = None) -> int:
        """Get the depth of the tree."""
        if node is None:
            node = self.root

        if node.value is not None:
            return 0

        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))


# ============================================================================
# K-MEANS CLUSTERING FROM SCRATCH
# ============================================================================

class KMeansScratch:
    """
    K-Means clustering implementation from scratch.

    Key concepts:
    - K-means++ initialization
    - Lloyd's algorithm
    - Elbow method for choosing k
    """

    def __init__(self, n_clusters: int = 3, max_iters: int = 100,
                 init: str = 'k-means++', random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init  # 'k-means++' or 'random'
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0

    def initialize_centroids_random(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids randomly."""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[random_indices].copy()

    def initialize_centroids_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using k-means++ algorithm.
        This gives better initial centroids than random initialization.
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Choose first centroid randomly
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            # Calculate distances to nearest centroid for each point
            distances = np.array([
                min([np.linalg.norm(x - c) for c in centroids]) for x in X
            ])

            # Choose next centroid with probability proportional to distance²
            probabilities = distances ** 2
            probabilities /= probabilities.sum()

            # Select next centroid
            cumsum = probabilities.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumsum):
                if r < p:
                    centroids.append(X[j])
                    break

        return np.array(centroids)

    def assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)

        return np.argmin(distances, axis=1)

    def update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids as the mean of assigned points."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster by reinitializing
                centroids[k] = X[np.random.randint(X.shape[0])]

        return centroids

    def fit(self, X: np.ndarray):
        """Fit K-means clustering."""
        # Initialize centroids
        if self.init == 'k-means++':
            self.centroids = self.initialize_centroids_kmeans_plus_plus(X)
        else:
            self.centroids = self.initialize_centroids_random(X)

        for iteration in range(self.max_iters):
            # Assign clusters
            labels = self.assign_clusters(X)

            # Update centroids
            new_centroids = self.update_centroids(X, labels)

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {iteration}")
                self.n_iter_ = iteration
                break

            self.centroids = new_centroids
            self.labels = labels

        # Calculate final inertia
        self.inertia_ = self.calculate_inertia(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        return self.assign_clusters(X)

    def calculate_inertia(self, X: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (WCSS).
        Lower is better.
        """
        distances = 0
        labels = self.assign_clusters(X)

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                distances += np.sum((cluster_points - self.centroids[k]) ** 2)

        return distances

    def silhouette_score(self, X: np.ndarray) -> float:
        """
        Calculate silhouette score.
        Ranges from -1 to 1, higher is better.
        """
        labels = self.assign_clusters(X)
        n_samples = X.shape[0]
        silhouette_scores = []

        for i in range(n_samples):
            # Current point and cluster
            point = X[i]
            cluster = labels[i]

            # Calculate a(i): mean distance to points in same cluster
            same_cluster_points = X[labels == cluster]
            if len(same_cluster_points) > 1:
                a_i = np.mean([np.linalg.norm(point - p) for p in same_cluster_points
                               if not np.array_equal(p, point)])
            else:
                a_i = 0

            # Calculate b(i): mean distance to points in nearest cluster
            b_i = np.inf
            for k in range(self.n_clusters):
                if k != cluster:
                    other_cluster_points = X[labels == k]
                    if len(other_cluster_points) > 0:
                        mean_dist = np.mean([np.linalg.norm(point - p)
                                             for p in other_cluster_points])
                        b_i = min(b_i, mean_dist)

            # Silhouette coefficient for this point
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0

            silhouette_scores.append(s_i)

        return np.mean(silhouette_scores)


def elbow_method_kmeans(X: np.ndarray, max_k: int = 10):
    """
    Elbow method to find optimal number of clusters.
    Plot WCSS vs number of clusters.
    """
    print("\n=== Elbow Method for K-Means ===")

    wcss = []
    silhouette_scores = []
    K = range(2, max_k + 1)

    for k in K:
        kmeans = KMeansScratch(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(kmeans.silhouette_score(X))
        print(f"k={k}: WCSS={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow plot
    ax1.plot(K, wcss, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Within-Cluster Sum of Squares')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    ax2.plot(K, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)

    # Find elbow point (using second derivative)
    if len(wcss) > 2:
        # Calculate second derivative
        second_derivative = np.diff(np.diff(wcss))
        elbow_idx = np.argmax(second_derivative) + 2  # +2 because of double diff
        elbow_k = list(K)[elbow_idx]
        ax1.axvline(x=elbow_k, color='g', linestyle='--',
                    label=f'Elbow at k={elbow_k}')
        ax1.legend()

    # Mark best silhouette
    best_silhouette_k = list(K)[np.argmax(silhouette_scores)]
    ax2.axvline(x=best_silhouette_k, color='g', linestyle='--',
                label=f'Best at k={best_silhouette_k}')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def visualize_clustering(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                         title: str = "K-Means Clustering"):
    """Visualize clustering results for 2D data."""
    if X.shape[1] != 2:
        print("Visualization requires 2D data. Using first two dimensions.")
        X = X[:, :2]
        centroids = centroids[:, :2]

    plt.figure(figsize=(10, 8))

    # Plot points
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[color], alpha=0.6, edgecolors='black',
                    linewidth=0.5, s=50, label=f'Cluster {label}')

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='red', s=300, marker='X', edgecolors='black',
                linewidth=2, label='Centroids')

    # Draw circles to show cluster boundaries
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            # Calculate radius as max distance from centroid
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            radius = np.percentile(distances, 90)  # 90th percentile
            circle = plt.Circle(centroid, radius, fill=False,
                                edgecolor=colors[i], linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


# ============================================================================
# CROSS-VALIDATION AND MODEL SELECTION
# ============================================================================

def manual_cross_validation(X: np.ndarray, y: np.ndarray, model_class,
                            k: int = 5, random_state: int = 42,
                            model_params: dict = None) -> Dict:
    """
    Implement k-fold cross-validation manually.
    Shows understanding of CV process for interviews.
    """
    print(f"\n=== {k}-Fold Cross-Validation ===")

    if model_params is None:
        model_params = {}

    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Store results
    train_scores = []
    val_scores = []
    models = []

    fold = 1
    for train_idx, val_idx in kf.split(X):
        print(f"\nFold {fold}/{k}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Calculate scores (accuracy for classification, R² for regression)
        if hasattr(model, 'predict_proba'):  # Classification
            train_score = np.mean(y_train == y_train_pred)
            val_score = np.mean(y_val == y_val_pred)
        else:  # Regression
            train_score = model.r2_score(y_train, y_train_pred)
            val_score = model.r2_score(y_val, y_val_pred)

        train_scores.append(train_score)
        val_scores.append(val_score)
        models.append(model)

        print(f"Train score: {train_score:.4f}")
        print(f"Val score: {val_score:.4f}")

        fold += 1

    # Calculate statistics
    results = {
        'train_scores': np.array(train_scores),
        'val_scores': np.array(val_scores),
        'mean_train_score': np.mean(train_scores),
        'std_train_score': np.std(train_scores),
        'mean_val_score': np.mean(val_scores),
        'std_val_score': np.std(val_scores),
        'models': models
    }

    print(f"\n=== Cross-Validation Summary ===")
    print(f"Mean train score: {results['mean_train_score']:.4f} (+/- {results['std_train_score']:.4f})")
    print(f"Mean val score: {results['mean_val_score']:.4f} (+/- {results['std_val_score']:.4f})")

    # Visualize CV results
    plt.figure(figsize=(10, 6))

    folds = range(1, k + 1)
    plt.plot(folds, train_scores, 'bo-', label='Train scores', linewidth=2, markersize=8)
    plt.plot(folds, val_scores, 'ro-', label='Validation scores', linewidth=2, markersize=8)

    # Add mean lines
    plt.axhline(y=results['mean_train_score'], color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=results['mean_val_score'], color='red', linestyle='--', alpha=0.5)

    # Add confidence intervals
    plt.fill_between(folds,
                     results['mean_train_score'] - results['std_train_score'],
                     results['mean_train_score'] + results['std_train_score'],
                     alpha=0.2, color='blue')
    plt.fill_between(folds,
                     results['mean_val_score'] - results['std_val_score'],
                     results['mean_val_score'] + results['std_val_score'],
                     alpha=0.2, color='red')

    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title(f'{k}-Fold Cross-Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all machine learning demonstrations."""
    print("=" * 60)
    print("DATA SCIENCE INTERVIEW PREP - MACHINE LEARNING")
    print("=" * 60)

    # Set random seed
    np.random.seed(42)

    # 1. Linear Regression
    print("\n1. LINEAR REGRESSION")
    print("-" * 40)
    compare_linear_regression_methods()

    # Show convergence
    lr = LinearRegressionScratch(learning_rate=0.1, n_iterations=100)
    X = np.random.randn(100, 3)
    y = X @ np.array([2, -1, 0.5]) + 3 + np.random.randn(100) * 0.1
    lr.fit(X, y, verbose=False)
    lr.plot_convergence()

    # 2. Logistic Regression
    print("\n\n2. LOGISTIC REGRESSION")
    print("-" * 40)

    # Generate classification data
    X_class, y_class = make_classification(n_samples=200, n_features=2,
                                           n_redundant=0, n_informative=2,
                                           n_clusters_per_class=1, random_state=42)

    # Fit logistic regression
    lr_class = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    lr_class.fit(X_class, y_class, verbose=False)

    # Visualize decision boundary
    lr_class.decision_boundary(X_class, y_class)

    # 3. Bias-Variance Tradeoff
    print("\n\n3. BIAS-VARIANCE TRADEOFF")
    print("-" * 40)
    visualize_bias_variance_tradeoff()

    # 4. Classification Metrics
    print("\n\n4. CLASSIFICATION METRICS")
    print("-" * 40)

    # Evaluate logistic regression
    y_pred = lr_class.predict(X_class)
    y_proba = lr_class.predict_proba(X_class)
    metrics = calculate_classification_metrics(y_class, y_pred, y_proba)
    plot_classification_metrics(metrics)

    # 5. Decision Trees
    print("\n\n5. DECISION TREES")
    print("-" * 40)

    # Generate data for decision tree
    X_tree, y_tree = make_classification(n_samples=150, n_features=4,
                                         n_informative=3, n_redundant=1,
                                         random_state=42)

    # Train decision tree
    dt = DecisionTreeClassifierScratch(max_depth=3, min_samples_split=5)
    dt.fit(X_tree, y_tree)

    print("\nDecision Tree Structure:")
    dt.print_tree()

    print(f"\nTree depth: {dt.get_depth()}")
    print(f"Feature importances: {dt.feature_importances_}")

    # Evaluate
    y_pred_tree = dt.predict(X_tree)
    accuracy = np.mean(y_pred_tree == y_tree)
    print(f"Training accuracy: {accuracy:.4f}")

    # 6. K-Means Clustering
    print("\n\n6. K-MEANS CLUSTERING")
    print("-" * 40)

    # Generate clustering data
    X_cluster, y_true = make_blobs(n_samples=300, n_features=2, centers=4,
                                   cluster_std=1.0, random_state=42)

    # Elbow method
    elbow_method_kmeans(X_cluster, max_k=8)

    # Fit K-means with optimal k
    kmeans = KMeansScratch(n_clusters=4, init='k-means++')
    kmeans.fit(X_cluster)

    print(f"\nFinal inertia: {kmeans.inertia_:.2f}")
    print(f"Silhouette score: {kmeans.silhouette_score(X_cluster):.3f}")

    # Visualize results
    visualize_clustering(X_cluster, kmeans.labels, kmeans.centroids)

    # 7. Cross-Validation
    print("\n\n7. CROSS-VALIDATION")
    print("-" * 40)

    # Cross-validate logistic regression
    cv_results = manual_cross_validation(X_class, y_class,
                                         LogisticRegressionScratch,
                                         k=5, model_params={'learning_rate': 0.1})

    print("\n" + "=" * 60)
    print("MACHINE LEARNING DEMONSTRATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Run demonstrations
    main()