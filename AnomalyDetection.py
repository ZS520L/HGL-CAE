import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Define the HGL model
class HGL(nn.Module):
    def __init__(self, dim=4, units=1, sef=1, init_mode='randn'):
        super(HGL, self).__init__()

        if init_mode in ['normal', 0, '0']:
            self.mean = nn.Parameter(torch.zeros(dim * units))
        elif init_mode in ['randn', 1, '1']:
            self.mean = nn.Parameter(torch.randn(dim * units))

        self.sigma = nn.Parameter(torch.ones(dim * units) * sef)
        self.dim = dim
        self.units = units
        self.sef = sef

    def forward(self, din):
        sigma = torch.abs(self.sigma)

        # Generate Gaussian distribution
        normal = Normal(self.mean, sigma)

        # Expand input data to match the number of units
        din = din.repeat(1, self.units)

        # Compute log probabilities and exponentiate to get probability densities
        din = normal.log_prob(din).exp()

        # Reshape and compute joint probability density
        din = din.reshape(din.shape[0], self.units, self.dim)
        din = torch.prod(din, dim=-1)

        # Return the pooled result and corresponding indices
        return torch.max(din, dim=1, keepdim=True)

# Load the Iris dataset
iris = load_iris()
X_numpy = iris.data
labels_true = iris.target

# Split data into normal (classes 0 and 1) and anomalies (class 2)
normal_classes = [0, 1]
anomaly_class = 2

# Select normal data (classes 0 and 1)
normal_indices = np.isin(labels_true, normal_classes)
X_normal = X_numpy[normal_indices]
labels_normal = labels_true[normal_indices]

# Select anomaly data (class 2)
anomaly_indices = labels_true == anomaly_class
X_anomaly = X_numpy[anomaly_indices]
labels_anomaly = labels_true[anomaly_indices]

# Data preprocessing
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_anomaly_scaled = scaler.transform(X_anomaly)

# Convert data to tensors
X_normal_tensor = torch.from_numpy(X_normal_scaled.astype(np.float32))
X_anomaly_tensor = torch.from_numpy(X_anomaly_scaled.astype(np.float32))

# Set HGL model parameters
dim = X_numpy.shape[1]
units = 4  # Modeling normal data with a single Gaussian
sef = 1
init_mode = 'normal'

# Initialize the HGL model
hgl = HGL(dim=dim, units=units, sef=sef, init_mode=init_mode)

# Define optimizer and loss function
optimizer = torch.optim.Adam(hgl.parameters(), lr=0.01)
n_epochs = 1000

# Training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Forward pass with normal data
    prob, _ = hgl(X_normal_tensor)
    
    # Compute negative log-likelihood loss
    log_prob = torch.log(prob + 1e-8)
    loss = -torch.mean(log_prob)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on normal and anomaly data
with torch.no_grad():
    # Compute probabilities for normal data
    prob_normal, _ = hgl(X_normal_tensor)
    log_prob_normal = torch.log(prob_normal + 1e-8).numpy().flatten()
    
    # Compute probabilities for anomaly data
    prob_anomaly, _ = hgl(X_anomaly_tensor)
    log_prob_anomaly = torch.log(prob_anomaly + 1e-8).numpy().flatten()

# Plot histograms of log probabilities
plt.figure(figsize=(10, 5))
plt.hist(log_prob_normal, bins=30, alpha=0.5, label='Normal Data')
plt.hist(log_prob_anomaly, bins=30, alpha=0.5, label='Anomaly Data')
plt.legend()
plt.xlabel('Log Probability')
plt.ylabel('Frequency')
plt.title('Log Probabilities of Normal and Anomaly Data')
plt.show()

# Compute ROC curve and AUC
log_probs = np.concatenate([log_prob_normal, log_prob_anomaly])
labels = np.concatenate([np.zeros_like(log_prob_normal), np.ones_like(log_prob_anomaly)])  # 0: normal, 1: anomaly

fpr, tpr, thresholds = roc_curve(labels, -log_probs)  # Negative because higher log_prob indicates normal
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Anomaly Detection')
plt.legend(loc="lower right")
plt.show()

print(f'Area Under Curve (AUC): {roc_auc:.4f}')
