#!/usr/bin/env python3
"""
Data Science Interview Prep - Machine Learning
=============================================
Key algorithms and common interview questions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

# ============================================================================
# LINEAR REGRESSION
# ============================================================================

class LinearRegression:
    """Simple linear regression implementation"""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.epochs):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Gradients
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias

# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================

class LogisticRegression:
    """Simple logistic regression implementation"""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """Train the model"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.epochs):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            # Gradients
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        return (self.predict_proba(X) >= threshold).astype(int)

# ============================================================================
# DECISION TREE
# ============================================================================

class DecisionTree:
    """Simple decision tree for classification"""
    
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def gini_impurity(self, y):
        """Calculate Gini impurity"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def find_best_split(self, X, y):
        """Find the best split point"""
        best_gini = 1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_gini = self.gini_impurity(y[left_mask])
                right_gini = self.gini_impurity(y[right_mask])
                
                # Weighted average
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)
                
                gini = (n_left * left_gini + n_right * right_gini) / n_total
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the tree"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < 2 or 
            n_classes == 1):
            return {'prediction': np.argmax(np.bincount(y))}
        
        # Find best split
        feature, threshold = self.find_best_split(X, y)
        
        if feature is None:
            return {'prediction': np.argmax(np.bincount(y))}
        
        # Create split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Recursively build children
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Train the tree"""
        self.tree = self.build_tree(X, y)
    
    def predict_one(self, x, tree):
        """Predict for a single sample"""
        if 'prediction' in tree:
            return tree['prediction']
        
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_one(x, tree['left'])
        else:
            return self.predict_one(x, tree['right'])
    
    def predict(self, X):
        """Make predictions"""
        return np.array([self.predict_one(x, self.tree) for x in X])

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

class KMeans:
    """Simple K-means clustering"""
    
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        """Fit K-means to the data"""
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) 
                                     for k in range(self.n_clusters)])
            
            # Check convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
    
    def predict(self, X):
        """Assign new points to clusters"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# ============================================================================
# COMMON INTERVIEW QUESTIONS
# ============================================================================

def explain_bias_variance():
    """Explain bias-variance tradeoff"""
    print("=== Bias-Variance Tradeoff ===")
    print("Bias: Error due to overly simple model")
    print("Variance: Error due to model being too complex")
    print("Goal: Find sweet spot between underfitting and overfitting")
    print()

def explain_cross_validation():
    """Explain cross-validation"""
    print("=== Cross-Validation ===")
    print("K-fold CV: Split data into K parts")
    print("Train on K-1 parts, validate on 1 part")
    print("Repeat K times, average results")
    print("Helps estimate model performance")
    print()

def explain_regularization():
    """Explain regularization"""
    print("=== Regularization ===")
    print("L1 (Lasso): Adds |w| penalty, creates sparse models")
    print("L2 (Ridge): Adds w² penalty, prevents large weights")
    print("Elastic Net: Combines L1 and L2")
    print()

def demonstrate_algorithms():
    """Demonstrate all algorithms"""
    print("=== Algorithm Demonstrations ===")
    
    # Generate data
    np.random.seed(42)
    X_reg, y_reg = make_regression(n_samples=100, n_features=2, noise=0.1)
    X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1)
    
    # Linear Regression
    print("1. Linear Regression:")
    lr = LinearRegression()
    lr.fit(X_reg, y_reg)
    predictions = lr.predict(X_reg)
    mse = np.mean((predictions - y_reg) ** 2)
    print(f"   MSE: {mse:.4f}")
    
    # Logistic Regression
    print("2. Logistic Regression:")
    log_reg = LogisticRegression()
    log_reg.fit(X_clf, y_clf)
    predictions = log_reg.predict(X_clf)
    accuracy = np.mean(predictions == y_clf)
    print(f"   Accuracy: {accuracy:.4f}")
    
    # Decision Tree
    print("3. Decision Tree:")
    dt = DecisionTree(max_depth=3)
    dt.fit(X_clf, y_clf)
    predictions = dt.predict(X_clf)
    accuracy = np.mean(predictions == y_clf)
    print(f"   Accuracy: {accuracy:.4f}")
    
    # K-Means
    print("4. K-Means Clustering:")
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_clf)
    print(f"   Centroids shape: {kmeans.centroids.shape}")
    print()

def main():
    """Run all demonstrations"""
    explain_bias_variance()
    explain_cross_validation()
    explain_regularization()
    demonstrate_algorithms()

if __name__ == "__main__":
    main()