# 🧠 Diabetes Prediction using Artificial Neural Network (ANN)

This project aims to predict whether a person has diabetes based on medical attributes using an **Artificial Neural Network (ANN)**. The model is trained on the **Pima Indians Diabetes Dataset**.
link:https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data
---

## 📊 Dataset: Pima Indians Diabetes Dataset
- The dataset consists of **768 samples** with **8 features** related to diabetes diagnosis.
- **Features:**
  - `Pregnancies`: Number of times pregnant
  - `Glucose`: Plasma glucose concentration
  - `BloodPressure`: Diastolic blood pressure (mm Hg)
  - `SkinThickness`: Triceps skin fold thickness (mm)
  - `Insulin`: 2-Hour serum insulin (mu U/ml)
  - `BMI`: Body mass index (weight in kg/(height in m)^2)
  - `DiabetesPedigreeFunction`: Diabetes pedigree function
  - `Age`: Age of the person
  - `Outcome`: (0 = No Diabetes, 1 = Diabetes)

---

## 🚀 Project Workflow

1. **Data Preprocessing**
   - Load dataset using `pandas`
   - Perform **feature scaling** using `StandardScaler`
   - Split dataset into **training (80%)** and **testing (20%)** sets

2. **Building the Artificial Neural Network (ANN)**
   - Input layer: **8 features**
   - Hidden Layer 1: **32 neurons, ReLU activation**
   - Hidden Layer 2: **32 neurons, ReLU activation**
   - Output Layer: **1 neuron (Sigmoid activation for binary classification)**

3. **Model Compilation & Training**
   - Loss Function: **Binary Cross-Entropy**
   - Optimizer: **Adam**
   - Evaluation: **Accuracy on test data**

4. **Model Evaluation**
   - Compute **Confusion Matrix**
   - Calculate **Accuracy, Precision, Recall, and F1-Score**

---

## 📌 Key Results

| Metric  | Score |
|---------|-------|
| Training Accuracy | ~90% |
| Test Accuracy | ~78% |
| Confusion Matrix | `[[92 15] [18 29]]` |
| Potential Overfitting | **Yes, accuracy drop from train to test (~12%)** |

🔹 **Precision, Recall, and F1-score analysis is required to check False Negatives (FN) and False Positives (FP).**  
🔹 **To improve performance, we can apply dropout, L2 regularization, and hyperparameter tuning.**

---

## 🛠️ Installation & Setup

### 🔹 **Step 1: Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/diabetes-prediction-ann.git
cd diabetes-prediction-ann
