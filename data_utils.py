import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_data(n_samples=1000, random_seed=42):
    """
    Generate a simple binary classification dataset with two clusters.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        The number of samples to generate.
    random_seed : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        The input features.
    y : ndarray of shape (n_samples,)
        The target labels (0 or 1).
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Initialize arrays
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples)
    
    # First cluster (class 0)
    X[:n_samples//2, 0] = np.random.normal(loc=-2, scale=1, size=n_samples//2)
    X[:n_samples//2, 1] = np.random.normal(loc=-2, scale=1, size=n_samples//2)
    y[:n_samples//2] = 0
    
    # Second cluster (class 1)
    X[n_samples//2:, 0] = np.random.normal(loc=2, scale=1, size=n_samples//2)
    X[n_samples//2:, 1] = np.random.normal(loc=2, scale=1, size=n_samples//2)
    y[n_samples//2:] = 1
    
    return X, y

def load_data(n_samples=1000, test_size=0.2, random_seed=42):
    """
    Generate and split data into training and testing sets.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        The number of samples to generate.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_seed : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    X_train : ndarray
        Training features.
    X_test : ndarray
        Testing features.
    y_train : ndarray
        Training labels.
    y_test : ndarray
        Testing labels.
    """
    # Generate data
    X, y = generate_data(n_samples, random_seed)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    
    return X_train, X_test, y_train, y_test

def plot_data(X, y, title="Binary Classification Data"):
    """
    Plot the binary classification data.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, 2)
        The input features.
    y : ndarray of shape (n_samples,)
        The target labels.
    title : str, default="Binary Classification Data"
        The plot title.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_decision_boundary(X, y, model):
    """
    Plot the decision boundary of a trained model.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, 2)
        The input features.
    y : ndarray of shape (n_samples,)
        The target labels.
    model : keras.Model
        The trained model.
    """
    # Define the grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_training_history(history):
    """
    Plot the training history of a model.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        The history object returned by model.fit().
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    X, y = generate_data(1000)
    plot_data(X, y)
    print(f"Generated {len(X)} samples with shape {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
