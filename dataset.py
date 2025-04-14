# generate_dataset.py
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples per class (balanced)
samples_per_class = 10000

def generate_class_data(label, mean_age, mean_bmi, mean_bp, mean_glucose):
    age = np.random.normal(loc=mean_age, scale=5, size=samples_per_class)
    bmi = np.random.normal(loc=mean_bmi, scale=3, size=samples_per_class)
    bp = np.random.normal(loc=mean_bp, scale=8, size=samples_per_class)  # blood pressure
    glucose = np.random.normal(loc=mean_glucose, scale=15, size=samples_per_class)
    return pd.DataFrame({
        'Age': age,
        'BMI': bmi,
        'BloodPressure': bp,
        'Glucose': glucose,
        'Label': label
    })

# Generate synthetic data for each class:
# Healthy (Label 0): parameters in normal range.
data_healthy = generate_class_data(label=0, mean_age=45, mean_bmi=24, mean_bp=115, mean_glucose=90)
# Diabetes (Label 1): higher BMI and glucose.
data_diabetes = generate_class_data(label=1, mean_age=50, mean_bmi=30, mean_bp=120, mean_glucose=150)
# Heart Disease (Label 2): higher blood pressure.
data_heart = generate_class_data(label=2, mean_age=55, mean_bmi=28, mean_bp=140, mean_glucose=110)

# Combine the datasets into a single DataFrame
data = pd.concat([data_healthy, data_diabetes, data_heart], ignore_index=True)
print("Generated dataset shape:", data.shape)

# Save the dataset to CSV
data.to_csv('generatedataset.csv', index=False)
print("Dataset saved to 'generatedataset.csv'")
