import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = loadmat('dataset.mat')
X = data['data_X']  # Shape: (100, 3)
D = data['target_D']  # Shape: (100, 1)

def compute_wiener_solution(X, D):
    """
    Compute the optimal Wiener solution using the normal equation: W* = (X^T X)^(-1) X^T D
    """
    return np.linalg.inv(X.T @ X) @ X.T @ D

def compute_mse_loss(X, D, W):
    """
    Compute Mean Squared Error loss for given weights
    """
    y_pred = X @ W
    return np.mean((D - y_pred) ** 2)

def train_lms(X, D, learning_rate, epochs, W_init):
    """
    Train using LMS algorithm (online learning)
    Returns:
        - Final weights
        - History of MSE loss after each epoch
    """
    W = W_init.copy()
    n_samples = X.shape[0]
    loss_history = []
    
    for epoch in range(epochs):
        # Shuffle data for each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        D_shuffled = D[indices]
        
        # Process one sample at a time
        for i in range(n_samples):
            x_i = X_shuffled[i:i+1].T  # Make column vector
            d_i = D_shuffled[i:i+1].T  # Make column vector
            
            # Compute prediction
            y_i = X_shuffled[i:i+1] @ W
            
            # Compute error
            e_i = d_i - y_i
            
            # Update weights using LMS rule
            W += learning_rate * x_i * e_i
        
        # Record MSE loss after each epoch
        current_mse = compute_mse_loss(X, D, W)
        loss_history.append(current_mse)
    
    return W, loss_history

# Part a: Compute Wiener solution
W_optimal = compute_wiener_solution(X, D)
optimal_mse = compute_mse_loss(X, D, W_optimal)

print("Optimal Wiener solution:")
print(f"W* = {W_optimal.flatten()}")
print(f"Optimal MSE = {optimal_mse}")

# Part b: Train using LMS with learning rate 0.005
W_init = np.array([[0.53], [0.20], [0.10]])
epochs = 20
learning_rate = 0.005

W_lms, loss_history = train_lms(X, D, learning_rate, epochs, W_init)

# Plot MSE loss vs Epochs
plt.figure(figsize=(10, 6))
plt.semilogy(range(epochs), loss_history)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss (log scale)')
plt.title('LMS Training Progress')
plt.show()

# Part c: 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(X[:, 1], X[:, 2], D[:, 0], c='b', marker='o', label='Data points')

# Create mesh grid for plotting planes
x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min, x1_max, 20),
                              np.linspace(x2_min, x2_max, 20))

# Plot optimal plane
Z_optimal = W_optimal[0] + W_optimal[1]*x1_grid + W_optimal[2]*x2_grid
ax.plot_surface(x1_grid, x2_grid, Z_optimal, alpha=0.3, color='r', label='Optimal solution')

# Plot LMS plane
Z_lms = W_lms[0] + W_lms[1]*x1_grid + W_lms[2]*x2_grid
ax.plot_surface(x1_grid, x2_grid, Z_lms, alpha=0.3, color='g', label='LMS solution')

ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('d')
plt.title('Data Points and Fitted Planes')
plt.show()

# Part d: Compare different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.5]
results = []

plt.figure(figsize=(12, 8))
for lr in learning_rates:
    W_temp, loss_history = train_lms(X, D, lr, epochs, W_init)
    plt.semilogy(range(epochs), loss_history, label=f'lr = {lr}')

plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss (log scale)')
plt.title('LMS Training Progress with Different Learning Rates')
plt.legend()
plt.show()

# Test extreme learning rate
lr_extreme = 1.0
W_extreme, loss_history_extreme = train_lms(X, D, lr_extreme, epochs, W_init)
plt.figure(figsize=(10, 6))
plt.semilogy(range(epochs), loss_history_extreme)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss (log scale)')
plt.title('LMS Training Progress with Learning Rate = 1.0')
plt.show()