import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Generate some sample data (a simple binary classification problem)
def generate_data(n_samples=1000):
    # Create two clusters of points
    np.random.seed(42)
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

# Create the data
X, y = generate_data(1000)

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(2,)),  # Hidden layer with 10 neurons
    layers.Dense(1, activation='sigmoid')                   # Output layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Plot the decision boundary
def plot_decision_boundary(X, y, model):
    # Define the grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# Visualize the results
plot_decision_boundary(X, y, model)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
