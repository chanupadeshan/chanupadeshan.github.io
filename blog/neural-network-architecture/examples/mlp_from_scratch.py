"""A small neural network written with NumPy for learning purposes."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


rng = np.random.default_rng(42)
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])

# Architecture: 2 inputs -> 4 ReLU units -> 1 sigmoid output.
W1 = rng.normal(0, np.sqrt(2 / 2), size=(2, 4))
b1 = np.zeros((1, 4))
W2 = rng.normal(0, np.sqrt(2 / 4), size=(4, 1))
b2 = np.zeros((1, 1))

learning_rate = 0.1
for step in range(10_000):
    # Forward pass. Shapes: (4,2)@(2,4)->(4,4)@(4,1)->(4,1)
    z1 = X @ W1 + b1
    h1 = np.maximum(0, z1)
    logits = h1 @ W2 + b2
    predictions = sigmoid(logits)

    # For sigmoid plus binary cross-entropy, dL/dlogits = prediction - target.
    d_logits = (predictions - y) / len(X)
    dW2 = h1.T @ d_logits
    db2 = d_logits.sum(axis=0, keepdims=True)
    d_hidden = d_logits @ W2.T
    d_z1 = d_hidden * (z1 > 0)
    dW1 = X.T @ d_z1
    db1 = d_z1.sum(axis=0, keepdims=True)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Run one final forward pass with the updated parameters.
z1 = X @ W1 + b1
h1 = np.maximum(0, z1)
predictions = sigmoid(h1 @ W2 + b2)
epsilon = 1e-12
loss = -np.mean(
    y * np.log(predictions + epsilon)
    + (1 - y) * np.log(1 - predictions + epsilon)
)
classes = (predictions >= 0.5).astype(int)
print("Architecture: 2 -> 4 -> 1")
print("Parameters:", W1.size + b1.size + W2.size + b2.size)
print("Loss:", round(float(loss), 6))
print("Probabilities:", np.round(predictions.ravel(), 4))
print("Predictions:", classes.ravel())
