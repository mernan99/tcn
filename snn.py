import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define ANN model
class ANNModel(nn.Module):
    def __init__(self, input_size, h1=256, h2=128, h3=64, output_size=7):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

torch.manual_seed(41)  # Reproducibility

#  .mat files 
folder_path = 'S1'
mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

X_all = []
y_all = []

samples_per_segment = 200
valid_labels = [1, 3, 4, 6, 9, 10, 11]
label_map = {label: idx for idx, label in enumerate(valid_labels)}

for file_name in mat_files:
    file_path = os.path.join(folder_path, file_name)
    data = loadmat(file_path)

    emg = data['emg']
    restimulus = data['restimulus'].flatten()

    change_indices = np.where(np.diff(restimulus.astype(int), prepend=0) > 0)[0]

    for idx in change_indices:
        label = int(restimulus[idx])
        if label not in label_map:
            continue
        if idx + samples_per_segment <= emg.shape[0]:
            window = emg[idx:idx + samples_per_segment]
            X_all.append(window.flatten())
            y_all.append(label_map[label])  # remapped label

X = np.array(X_all)
y = np.array(y_all)
output_size = len(valid_labels)

print(f"Loaded {len(X)} samples from {len(mat_files)} files.")
print("Label distribution:", np.bincount(y))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Normalize features (zero mean, unit variance using train stats)
mean = X_train.mean(dim=0)
std = X_train.std(dim=0) + 1e-6  # avoid division by zero

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std  # use train mean/std for test too

# Initialize model
model = ANNModel(input_size=X_train.shape[1], h1=64, h2=32, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
epochs = 100
losses = []

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot Loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Evaluate on test set
model.eval()
with torch.no_grad():
    y_eval = model(X_test)
    test_loss = criterion(y_eval, y_test)
    y_pred = y_eval.argmax(dim=1)
    accuracy = (y_pred == y_test).float().mean().item()

    print(f"\nTest Loss: {test_loss.item():.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Prediction distribution:", np.bincount(y_pred.numpy()))



# Print individual predictions
print("\nSample Predictions:")
with torch.no_grad():
    for i in range(26):
        data = X_test[i]
        y_val = model(data.unsqueeze(0))
        print(f"Predicted: {y_pred[i]}, True: {y_test[i]}")
        print(f"{i+1}.) {y_val.argmax(dim=1).item()}")

