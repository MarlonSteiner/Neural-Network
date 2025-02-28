# Simple Neural Network

This repository contains a basic implementation of a neural network for binary classification using TensorFlow and Keras.

## Description

This project demonstrates how to build, train and evaluate a simple neural network model with TensorFlow. It includes code to:

- Generate synthetic data for a binary classification problem
- Create a neural network with one hidden layer
- Train the model
- Evaluate its performance
- Visualize the decision boundary and training metrics

The neural network architecture is deliberately kept simple to serve as an introduction to deep learning concepts.

## Requirements

The code requires the following Python packages:
- numpy
- tensorflow
- scikit-learn
- matplotlib

You can install these dependencies using:
```
pip install -r requirements.txt
```

## Usage

Simply run the Python script:
```
python simple_neural_network.py
```

This will:
1. Generate the sample data
2. Build and train the model
3. Display the model's performance
4. Create visualizations of the decision boundary and training history

## Model Architecture

The neural network has the following structure:
- Input layer with 2 features
- Hidden layer with 10 neurons and ReLU activation
- Output layer with 1 neuron and sigmoid activation

## Example Output

The script will print the model summary and test accuracy, and display two visualizations:
1. A plot showing the decision boundary of the trained model
2. Training and validation metrics over time (loss and accuracy)

## License

[Add your preferred license here]

## Contact

[Add your contact information here]
