
# 🏥 Federated Learning for Personalized Healthcare  
**Privacy-Preserving Disease Prediction using Federated Machine Learning**

## 📌 Overview

This project demonstrates how **Federated Learning (FL)** can be applied to **healthcare** to predict diseases like **Diabetes** and **Heart Disease** using **privacy-preserving** machine learning. Instead of sending sensitive health data to a central server, FL trains models locally on simulated client devices and aggregates them to form a global model.

The project includes:
- **Synthetic Dataset Generation**
- **Federated Model Training (PyTorch)**
- **Streamlit App for Real-Time Predictions**

---

## 💡 Key Features

✅ Federated learning with 5 simulated clients  
✅ Predicts: **Healthy**, **Diabetes**, **Heart Disease**  
✅ Real-time prediction with a user-friendly interface  
✅ ~99% accuracy on synthetic data  
✅ All data generated locally – no privacy risk

---

## 📂 Project Structure

```bash
.
├── app.py                  # Streamlit app for predictions
├── trainmodel.py          # Federated learning model training
├── generate_dataset.py    # Synthetic health dataset generator
├── generatedataset.csv    # Generated dataset (auto-created)
├── global_model.pt        # Trained global model (auto-saved)
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation
```

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python generate_dataset.py
```

This will create a synthetic dataset with equal samples for:
- Healthy (Label 0)
- Diabetes (Label 1)
- Heart Disease (Label 2)

### 3. Train Federated Model

```bash
python trainmodel.py
```

This simulates federated learning over 5 clients and saves the trained model as `global_model.pt`.

### 4. Launch Prediction App

```bash
streamlit run app.py
```

Use the sliders/inputs to enter patient data and get a real-time disease prediction.

---

## 📊 Example Input

| Feature         | Example |
|----------------|---------|
| Age            | 52      |
| BMI            | 29.0    |
| Blood Pressure | 135     |
| Glucose        | 145     |

**Predicted Class:** Diabetes

---

## 📈 Accuracy

✅ The trained model reaches ~**99% accuracy** on test data (synthetic).  
📌 In a real-world deployment, actual clinical datasets should be used with privacy safeguards in place.

---

## 🔐 Why Federated Learning?

- ✅ Keeps **patient data private** (no need to centralize)
- ✅ Suitable for **IoT healthcare devices** and **mobile health apps**
- ✅ Enables **personalized care** without compromising confidentiality

---

## 📦 Requirements

```
numpy
pandas
torch
scikit-learn
streamlit
```

You can install them all using:

```bash
pip install -r requirements.txt
```

---

