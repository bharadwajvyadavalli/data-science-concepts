#!/usr/bin/env python3
"""
Data Science Interview Prep - Deep Learning
==========================================
Key concepts and common interview questions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid: 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    """Tanh: (e^x - e^(-x)) / (e^x + e^(-x))"""
    return np.tanh(x)

def softmax(x):
    """Softmax for multi-class classification"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ============================================================================
# SIMPLE NEURAL NETWORK
# ============================================================================

class SimpleNeuralNetwork:
    """Basic neural network for interview demonstrations"""
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases"""
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward propagation"""
        activations = [X]
        
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = relu(z)
            activations.append(a)
        
        # Output layer
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        a = sigmoid(z)
        activations.append(a)
        
        return activations
    
    def backward(self, X, y, activations):
        """Backward propagation (simplified)"""
        m = X.shape[0]
        
        # Output layer gradient
        dz = activations[-1] - y
        
        # Update weights and biases
        for i in range(len(self.weights) - 1, -1, -1):
            dw = (1/m) * activations[i].T @ dz
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
            
            if i > 0:
                dz = dz @ self.weights[i].T * (activations[i] > 0)  # ReLU derivative
    
    def train(self, X, y, epochs=100):
        """Train the network"""
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations)
            
            if epoch % 20 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        activations = self.forward(X)
        return activations[-1]

# ============================================================================
# COMMON INTERVIEW QUESTIONS
# ============================================================================

def explain_backpropagation():
    """Explain backpropagation concept"""
    print("=== Backpropagation Explanation ===")
    print("1. Forward pass: Compute predictions")
    print("2. Compute loss")
    print("3. Backward pass: Compute gradients using chain rule")
    print("4. Update parameters: w = w - learning_rate * gradient")
    print()

def explain_vanishing_gradients():
    """Explain vanishing gradient problem"""
    print("=== Vanishing Gradients ===")
    print("Problem: Gradients become very small in deep networks")
    print("Causes: Sigmoid/tanh activation functions")
    print("Solutions: ReLU, BatchNorm, Residual connections")
    print()

def explain_overfitting():
    """Explain overfitting and solutions"""
    print("=== Overfitting ===")
    print("Problem: Model memorizes training data")
    print("Solutions:")
    print("- Dropout")
    print("- Regularization (L1/L2)")
    print("- Data augmentation")
    print("- Early stopping")
    print()

def demonstrate_gradient_descent():
    """Simple gradient descent demonstration"""
    print("=== Gradient Descent Demo ===")
    
    # Simple function: f(x) = x^2 + 2
    def f(x): return x**2 + 2
    def df(x): return 2*x
    
    x = 3.0
    learning_rate = 0.1
    
    print(f"Starting point: x = {x}")
    for i in range(10):
        gradient = df(x)
        x = x - learning_rate * gradient
        print(f"Step {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")
    print()

# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

def xor_problem():
    """Solve XOR problem with neural network"""
    print("=== XOR Problem ===")
    
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create network: 2 -> 4 -> 1
    nn = SimpleNeuralNetwork([2, 4, 1], learning_rate=0.1)
    nn.train(X, y, epochs=1000)
    
    predictions = nn.predict(X)
    print("Predictions:")
    for i, (x, pred) in enumerate(zip(X, predictions)):
        print(f"Input: {x}, Predicted: {pred[0]:.3f}, Actual: {y[i][0]}")
    print()

def main():
    """Run all demonstrations"""
    explain_backpropagation()
    explain_vanishing_gradients()
    explain_overfitting()
    demonstrate_gradient_descent()
    xor_problem()

if __name__ == "__main__":
    main()