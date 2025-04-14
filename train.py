# trainmodel.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Load and Partition Dataset
# ---------------------------
# Load the dataset generated earlier
data = pd.read_csv('generatedataset.csv')
print("Dataset shape:", data.shape)

# Shuffle dataset
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Simulate 5 clients by splitting the data evenly
num_clients = 5
clients_data = []
client_size = len(data_shuffled) // num_clients

for i in range(num_clients):
    start = i * client_size
    end = (i + 1) * client_size if i < num_clients - 1 else len(data_shuffled)
    clients_data.append(data_shuffled.iloc[start:end].reset_index(drop=True))
    print(f"Client {i+1} data shape: {clients_data[i].shape}")

# ---------------------------
# 2. Define the Neural Network Model
# ---------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size=4, num_classes=3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ---------------------------
# 3. Utility Functions: Data Preparation, Training, and Evaluation
# ---------------------------
def prepare_data(client_df):
    # Extract features and labels
    X = client_df[['Age', 'BMI', 'BloodPressure', 'Glucose']].values.astype(np.float32)
    y = client_df['Label'].values.astype(np.int64)
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

def local_train(model, optimizer, criterion, data_loader, device):
    model.train()
    for features, labels in data_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Prepare DataLoaders for each client
client_loaders = [prepare_data(client) for client in clients_data]

# Create a global test loader using the entire dataset
X_all = data[['Age', 'BMI', 'BloodPressure', 'Glucose']].values.astype(np.float32)
y_all = data['Label'].values.astype(np.int64)
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)
X_all_tensor = torch.tensor(X_all)
y_all_tensor = torch.tensor(y_all)
all_dataset = TensorDataset(X_all_tensor, y_all_tensor)
test_loader = DataLoader(all_dataset, batch_size=64, shuffle=False)

# ---------------------------
# 4. Federated Learning Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = SimpleNN().to(device)

global_rounds = 200
local_epochs = 3
learning_rate = 0.05
criterion = nn.CrossEntropyLoss()

def federated_averaging(global_model, client_models, client_data_sizes):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        # Weighted average of each parameter
        global_dict[key] = sum(client_data_sizes[i] * client_models[i].state_dict()[key] for i in range(len(client_models))) / sum(client_data_sizes)
    global_model.load_state_dict(global_dict)
    return global_model

# ---------------------------
# 5. Federated Training Loop
# ---------------------------
for round_num in range(1, global_rounds + 1):
    print(f"\n--- Global Round {round_num} ---")
    client_models = []
    client_data_sizes = []
    
    for loader in client_loaders:
        # Each client initializes with the global model weights
        client_model = SimpleNN().to(device)
        client_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(client_model.parameters(), lr=learning_rate)
        
        # Train locally on each client's data
        for epoch in range(local_epochs):
            local_train(client_model, optimizer, criterion, loader, device)
        
        client_models.append(client_model)
        client_data_sizes.append(len(loader.dataset))
    
    # Perform Federated Averaging to update the global model
    global_model = federated_averaging(global_model, client_models, client_data_sizes)
    
    # Evaluate the global model on test data
    accuracy = evaluate(global_model, test_loader, device)
    print(f"Global Model Accuracy after round {round_num}: {accuracy:.4f}")

# Save the global model after training
torch.save(global_model.state_dict(), "global_model.pt")
print("Global model saved as 'global_model.pt'")
