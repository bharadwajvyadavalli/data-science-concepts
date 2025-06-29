#!/usr/bin/env python3
"""
Data Science Interview Prep - Deep Learning
==========================================
Neural network implementations from scratch and PyTorch examples
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Callable
import time
from tqdm import tqdm

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from torchvision import datasets, transforms

    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not installed. Some examples will be skipped.")
    TORCH_AVAILABLE = False


# ============================================================================
# NEURAL NETWORK FROM SCRATCH
# ============================================================================

class ActivationFunctions:
    """Collection of activation functions and their derivatives."""

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid (using activation value)"""
        return a * (1 - a)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """Tanh activation"""
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a: np.ndarray) -> np.ndarray:
        """Derivative of tanh (using activation value)"""
        return 1 - a ** 2

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Softmax activation for multi-class classification"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation"""
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU"""
        return np.where(z > 0, 1, alpha)


class NeuralNetworkScratch:
    """
    Multi-layer perceptron implementation from scratch.

    This implementation shows understanding of:
    - Forward propagation
    - Backpropagation algorithm
    - Gradient descent optimization
    - Various activation functions
    - Weight initialization strategies
    """

    def __init__(self, layer_sizes: List[int], activation: str = 'relu',
                 output_activation: str = 'softmax', learning_rate: float = 0.01,
                 weight_init: str = 'xavier'):
        """
        Args:
            layer_sizes: List of layer sizes including input and output
            activation: Hidden layer activation ('relu', 'sigmoid', 'tanh')
            output_activation: Output layer activation ('softmax', 'sigmoid')
            learning_rate: Learning rate for gradient descent
            weight_init: Weight initialization method ('xavier', 'he', 'random')
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()

        # Training history
        self.loss_history = []
        self.accuracy_history = []

    def _initialize_parameters(self):
        """Initialize weights using various strategies."""
        for i in range(self.n_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            if self.weight_init == 'xavier':
                # Xavier/Glorot initialization
                std = np.sqrt(2.0 / (input_size + output_size))
                w = np.random.randn(input_size, output_size) * std
            elif self.weight_init == 'he':
                # He initialization (good for ReLU)
                std = np.sqrt(2.0 / input_size)
                w = np.random.randn(input_size, output_size) * std
            else:
                # Random initialization
                w = np.random.randn(input_size, output_size) * 0.01

            b = np.zeros((1, output_size))

            self.weights.append(w)
            self.biases.append(b)

    def _get_activation_function(self, name: str) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative."""
        activations = {
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'leaky_relu': (ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative),
            'softmax': (ActivationFunctions.softmax, None)
        }
        return activations[name]

    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network.

        Returns:
            activations: List of activation values for each layer
            z_values: List of pre-activation values
        """
        activations = [X]
        z_values = []

        activation_func, _ = self._get_activation_function(self.activation)

        # Hidden layers
        for i in range(self.n_layers - 2):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            a = activation_func(z)
            activations.append(a)

        # Output layer
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_values.append(z)

        if self.output_activation == 'softmax':
            a = ActivationFunctions.softmax(z)
        elif self.output_activation == 'sigmoid':
            a = ActivationFunctions.sigmoid(z)
        else:
            a = z  # Linear output

        activations.append(a)

        return activations, z_values

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute loss based on output activation.

        Cross-entropy for softmax, binary cross-entropy for sigmoid.
        """
        epsilon = 1e-8  # Small value to prevent log(0)

        if self.output_activation == 'softmax':
            # Cross-entropy loss
            return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
        elif self.output_activation == 'sigmoid':
            # Binary cross-entropy
            return -np.mean(y_true * np.log(y_pred + epsilon) +
                            (1 - y_true) * np.log(1 - y_pred + epsilon))
        else:
            # Mean squared error
            return 0.5 * np.mean((y_true - y_pred) ** 2)

    def backward_propagation(self, X: np.ndarray, y: np.ndarray,
                             activations: List[np.ndarray],
                             z_values: List[np.ndarray]) -> Tuple[List, List]:
        """
        Backward pass to calculate gradients.

        Returns:
            dW: Weight gradients
            db: Bias gradients
        """
        m = X.shape[0]  # Number of samples

        dW = []
        db = []

        _, activation_derivative = self._get_activation_function(self.activation)

        # Output layer gradient
        if self.output_activation == 'softmax':
            # For softmax + cross-entropy, the gradient simplifies
            dz = activations[-1] - y
        elif self.output_activation == 'sigmoid':
            # For sigmoid + binary cross-entropy
            dz = activations[-1] - y
        else:
            # For linear output + MSE
            dz = activations[-1] - y

        # Backpropagate through layers
        for i in range(self.n_layers - 2, -1, -1):
            # Calculate gradients
            dw = (1 / m) * activations[i].T @ dz
            db = (1 / m) * np.sum(dz, axis=0, keepdims=True)

            dW.insert(0, dw)
            db.insert(0, db)

            if i > 0:
                # Backpropagate to previous layer
                da = dz @ self.weights[i].T
                dz = da * activation_derivative(z_values[i - 1])

        return dW, db

    def update_parameters(self, dW: List, db: List):
        """Update weights and biases using gradients."""
        for i in range(self.n_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            batch_size: int = 32, validation_data: Optional[Tuple] = None,
            verbose: bool = True):
        """
        Train the neural network.

        Args:
            X: Training data
            y: Training labels (one-hot encoded for multi-class)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_data: Optional (X_val, y_val) tuple
            verbose: Print training progress
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            epoch_loss = 0
            n_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward propagation
                activations, z_values = self.forward_propagation(X_batch)

                # Compute loss
                batch_loss = self.compute_loss(y_batch, activations[-1])
                epoch_loss += batch_loss
                n_batches += 1

                # Backward propagation
                dW, db = self.backward_propagation(X_batch, y_batch,
                                                   activations, z_values)

                # Update parameters
                self.update_parameters(dW, db)

            # Average epoch loss
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            # Calculate accuracy
            predictions = self.predict(X)
            if len(y.shape) > 1 and y.shape[1] > 1:  # Multi-class
                accuracy = np.mean(np.argmax(predictions, axis=1) ==
                                   np.argmax(y, axis=1))
            else:  # Binary
                accuracy = np.mean((predictions > 0.5) == y)

            self.accuracy_history.append(accuracy)

            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_predictions = self.predict(X_val)
                val_loss = self.compute_loss(y_val, val_predictions)

                if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                    val_accuracy = np.mean(np.argmax(val_predictions, axis=1) ==
                                           np.argmax(y_val, axis=1))
                else:
                    val_accuracy = np.mean((val_predictions > 0.5) == y_val)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, "
                      f"Accuracy: {accuracy:.4f}", end="")
                if validation_data is not None:
                    print(f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                else:
                    print()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        activations, _ = self.forward_propagation(X)
        return activations[-1]

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        predictions = self.predict(X)
        if self.output_activation == 'softmax':
            return np.argmax(predictions, axis=1)
        else:
            return (predictions > 0.5).astype(int).ravel()

    def plot_training_history(self):
        """Plot training loss and accuracy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss plot
        ax1.plot(self.loss_history, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(self.accuracy_history, linewidth=2, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.show()


def demonstrate_neural_network():
    """Demonstrate neural network on various datasets."""
    print("=== Neural Network from Scratch Demo ===")

    # 1. XOR Problem (classic non-linear problem)
    print("\n1. XOR Problem:")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    nn_xor = NeuralNetworkScratch(
        layer_sizes=[2, 4, 1],
        activation='sigmoid',
        output_activation='sigmoid',
        learning_rate=0.5
    )

    nn_xor.fit(X_xor, y_xor, epochs=1000, batch_size=4, verbose=False)

    predictions = nn_xor.predict(X_xor)
    print("XOR Truth Table:")
    for i, (input_val, pred) in enumerate(zip(X_xor, predictions)):
        print(f"{input_val} -> {pred[0]:.3f} (expected: {y_xor[i][0]})")

    # 2. Multi-class Classification
    print("\n2. Multi-class Classification:")
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_classes=3, n_clusters_per_class=1, random_state=42)

    # One-hot encode labels
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Train network
    nn_multiclass = NeuralNetworkScratch(
        layer_sizes=[20, 64, 32, 3],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.01,
        weight_init='he'
    )

    nn_multiclass.fit(X_train, y_train, epochs=100, batch_size=32,
                      validation_data=(X_test, y_test), verbose=True)

    # Evaluate
    test_predictions = nn_multiclass.predict_classes(X_test)
    test_accuracy = np.mean(test_predictions == np.argmax(y_test, axis=1))
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")

    # Plot training history
    nn_multiclass.plot_training_history()

    # 3. Visualize decision boundaries for 2D data
    print("\n3. 2D Classification Visualization:")
    X_2d, y_2d = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                     n_informative=2, n_clusters_per_class=1,
                                     n_classes=3, random_state=42)

    y_2d_onehot = OneHotEncoder(sparse_output=False).fit_transform(y_2d.reshape(-1, 1))

    nn_2d = NeuralNetworkScratch(
        layer_sizes=[2, 8, 3],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.1
    )

    nn_2d.fit(X_2d, y_2d_onehot, epochs=200, batch_size=32, verbose=False)

    # Plot decision boundaries
    plot_decision_boundaries(nn_2d, X_2d, y_2d)


def plot_decision_boundaries(model, X: np.ndarray, y: np.ndarray):
    """Plot decision boundaries for 2D classification."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                          edgecolors='black', s=50)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Neural Network Decision Boundaries')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# PYTORCH IMPLEMENTATION
# ============================================================================

if TORCH_AVAILABLE:

    class SimpleMLP(nn.Module):
        """Simple Multi-Layer Perceptron in PyTorch."""

        def __init__(self, input_size: int, hidden_sizes: List[int],
                     output_size: int, dropout_rate: float = 0.2):
            super(SimpleMLP, self).__init__()

            layers = []
            prev_size = input_size

            # Hidden layers
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_size = hidden_size

            # Output layer
            layers.append(nn.Linear(prev_size, output_size))

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)


    class SimpleCNN(nn.Module):
        """
        CNN for image classification (e.g., MNIST).

        Architecture:
        - Conv2d -> ReLU -> MaxPool2d
        - Conv2d -> ReLU -> MaxPool2d
        - Flatten -> Linear -> ReLU -> Dropout
        - Linear (output)
        """

        def __init__(self, num_classes: int = 10):
            super(SimpleCNN, self).__init__()

            # Convolutional layers
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)

            # Fully connected layers
            # For MNIST: after 2 pooling layers, 28x28 -> 7x7
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)

            # Regularization
            self.dropout = nn.Dropout(0.5)

            # Batch normalization
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)

        def forward(self, x):
            # First conv block
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool(x)

            # Second conv block
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool(x)

            # Flatten
            x = x.view(-1, 64 * 7 * 7)

            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x


    def train_pytorch_model(model, train_loader, test_loader, num_epochs: int = 10,
                            learning_rate: float = 0.001, device: str = 'cpu'):
        """Train a PyTorch model with proper training loop."""
        print(f"\n=== Training PyTorch Model on {device} ===")

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            # Progress bar
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(device), target.to(device)

                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * correct / total

            # Validation phase
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, target)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            avg_test_loss = test_loss / len(test_loader)
            test_accuracy = 100. * correct / total

            # Update scheduler
            scheduler.step(avg_test_loss)

            # Save history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['test_loss'].append(avg_test_loss)
            history['test_acc'].append(test_accuracy)

            print(f'Epoch [{epoch + 1}/{num_epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

        return history


    def plot_training_history_pytorch(history: Dict):
        """Plot PyTorch training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
        ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    class EarlyStopping:
        """
        Early stopping to prevent overfitting.

        Monitors validation loss and stops training when it stops improving.
        """

        def __init__(self, patience: int = 7, min_delta: float = 0,
                     restore_best_weights: bool = True, verbose: bool = True):
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best_weights = restore_best_weights
            self.verbose = verbose
            self.best_loss = None
            self.best_model = None
            self.counter = 0
            self.early_stop = False

        def __call__(self, val_loss: float, model):
            if self.best_loss is None:
                self.best_loss = val_loss
                self.save_checkpoint(model)
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.restore_best_weights and self.best_model is not None:
                        model.load_state_dict(self.best_model)
                        if self.verbose:
                            print('Restoring best model weights')
            else:
                self.best_loss = val_loss
                self.save_checkpoint(model)
                self.counter = 0

        def save_checkpoint(self, model):
            """Save model state."""
            if self.restore_best_weights:
                self.best_model = model.state_dict().copy()


    def save_checkpoint(model, optimizer, epoch: int, loss: float,
                        filepath: str = 'checkpoint.pth'):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)
        print(f"Checkpoint saved to {filepath}")


    def load_checkpoint(filepath: str, model, optimizer=None):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
        return model, optimizer, epoch, loss

# ============================================================================
# CUSTOM DATASET AND DATA AUGMENTATION
# ============================================================================

if TORCH_AVAILABLE:

    class CustomDataset(Dataset):
        """
        Example of custom PyTorch dataset.

        Shows how to create datasets for non-standard data.
        """

        def __init__(self, data: np.ndarray, labels: np.ndarray,
                     transform=None, target_transform=None):
            self.data = torch.FloatTensor(data)
            self.labels = torch.LongTensor(labels)
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]

            if self.transform:
                sample = self.transform(sample)

            if self.target_transform:
                label = self.target_transform(label)

            return sample, label


    def create_data_augmentation():
        """Create data augmentation pipeline for images."""
        # Training augmentation
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST stats
        ])

        # Test augmentation (only normalization)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        return train_transform, test_transform


# ============================================================================
# ADVANCED TECHNIQUES
# ============================================================================

def demonstrate_regularization_techniques():
    """Demonstrate various regularization techniques."""
    print("\n=== Regularization Techniques Demo ===")

    # Generate overfitting-prone data
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

    # One-hot encode for neural network
    y_onehot = np.eye(2)[y]

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.3, random_state=42
    )

    # 1. No regularization (baseline)
    print("\n1. No Regularization:")
    nn_base = NeuralNetworkScratch(
        layer_sizes=[2, 128, 64, 2],  # Intentionally large
        activation='relu',
        output_activation='softmax',
        learning_rate=0.01
    )
    nn_base.fit(X_train, y_train, epochs=200, verbose=False)

    # 2. L2 Regularization (Weight Decay)
    print("\n2. L2 Regularization:")
    # This would be implemented by adding penalty term to loss

    # 3. Dropout (demonstrated in PyTorch section)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model, title in zip(axes, [nn_base], ['No Regularization']):
        # Decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train.argmax(axis=1),
                   cmap='RdYlBu', edgecolors='black', s=50)
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()


def learning_rate_scheduling_demo():
    """Demonstrate different learning rate schedules."""
    print("\n=== Learning Rate Scheduling Demo ===")

    epochs = 100
    initial_lr = 0.1

    # Different scheduling strategies
    schedules = {
        'Constant': lambda epoch: initial_lr,
        'Step Decay': lambda epoch: initial_lr * (0.5 ** (epoch // 30)),
        'Exponential': lambda epoch: initial_lr * np.exp(-0.05 * epoch),
        'Cosine Annealing': lambda epoch: initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs)),
        '1/t Decay': lambda epoch: initial_lr / (1 + 0.1 * epoch)
    }

    plt.figure(figsize=(12, 8))

    for name, schedule in schedules.items():
        lrs = [schedule(epoch) for epoch in range(epochs)]
        plt.plot(lrs, label=name, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduling Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()


# ============================================================================
# MNIST DEMO WITH PYTORCH
# ============================================================================

def mnist_pytorch_demo():
    """Complete MNIST classification demo with PyTorch."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping MNIST demo.")
        return

    print("\n=== MNIST Classification with PyTorch ===")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Data loading with augmentation
    train_transform, test_transform = create_data_augmentation()

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SimpleCNN(num_classes=10)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    history = train_pytorch_model(
        model, train_loader, test_loader,
        num_epochs=num_epochs, learning_rate=learning_rate, device=device
    )

    # Plot training history
    plot_training_history_pytorch(history)

    # Visualize some predictions
    visualize_predictions(model, test_loader, device)


def visualize_predictions(model, test_loader, device):
    """Visualize model predictions on test data."""
    model.eval()

    # Get one batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Plot first 16 images with predictions
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.ravel()

    for idx in range(16):
        axes[idx].imshow(images[idx].cpu().squeeze(), cmap='gray')
        axes[idx].set_title(f'True: {labels[idx].item()}, Pred: {predicted[idx].item()}')
        axes[idx].axis('off')

        # Color code correct/incorrect
        if labels[idx] == predicted[idx]:
            axes[idx].set_title(f'True: {labels[idx].item()}, Pred: {predicted[idx].item()}',
                                color='green')
        else:
            axes[idx].set_title(f'True: {labels[idx].item()}, Pred: {predicted[idx].item()}',
                                color='red')

    plt.suptitle('MNIST Predictions (Green=Correct, Red=Incorrect)')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all deep learning demonstrations."""
    print("=" * 60)
    print("DATA SCIENCE INTERVIEW PREP - DEEP LEARNING")
    print("=" * 60)

    # Set random seeds
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    # 1. Neural Network from Scratch
    print("\n1. NEURAL NETWORK FROM SCRATCH")
    print("-" * 40)
    demonstrate_neural_network()

    # 2. Regularization Techniques
    print("\n\n2. REGULARIZATION TECHNIQUES")
    print("-" * 40)
    demonstrate_regularization_techniques()

    # 3. Learning Rate Scheduling
    print("\n\n3. LEARNING RATE SCHEDULING")
    print("-" * 40)
    learning_rate_scheduling_demo()

    # 4. PyTorch MNIST Demo
    if TORCH_AVAILABLE:
        print("\n\n4. PYTORCH MNIST DEMO")
        print("-" * 40)
        mnist_pytorch_demo()

    print("\n" + "=" * 60)
    print("DEEP LEARNING DEMONSTRATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Run demonstrations
    main()