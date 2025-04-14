# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Precomputed StandardScaler parameters from the training data
# (Adjust these values to match the training data statistics)
SCALER_MEANS = np.array([50.0, 27.0, 125.0, 117.0], dtype=np.float32)
SCALER_STDS = np.array([5.0, 3.0, 8.0, 15.0], dtype=np.float32)

def scale_input(x):
    """
    Normalize input data using the precomputed mean and standard deviation.
    """
    return (x - SCALER_MEANS) / SCALER_STDS

# Define the model structure (must match the structure used during training)
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

# Load the trained global model (assumed to have ~99% accuracy)
model = SimpleNN()
model.load_state_dict(torch.load("global_model.pt", map_location=torch.device('cpu')))
model.eval()

# Streamlit interface setup
st.title("Federated Learning - Disease Prediction")
st.write("This model has been tuned and demonstrates on test data.")

# Input form for patient data
age = st.number_input("Age", value=50, min_value=0, max_value=120)
bmi = st.number_input("BMI", value=27.0, min_value=10.0, max_value=50.0, step=0.1)
bp = st.number_input("Blood Pressure", value=125, min_value=50, max_value=200)
glucose = st.number_input("Glucose Level", value=117, min_value=50, max_value=300)

if st.button("Predict"):
    # Prepare the input array and normalize it using the precomputed scaler parameters
    input_data = np.array([[age, bmi, bp, glucose]], dtype=np.float32)
    input_scaled = scale_input(input_data)
    
    # Convert the scaled data to a tensor
    input_tensor = torch.tensor(input_scaled)
    
    # Perform the prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)
    
    # Map the model output to the class labels
    label_mapping = {0: "Healthy", 1: "Diabetes", 2: "Heart Disease"}
    st.write("Predicted Disease:", label_mapping[prediction.item()])
