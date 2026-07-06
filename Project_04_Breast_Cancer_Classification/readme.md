# Breast Cancer Classification using Logistic Regression

This project uses Machine Learning (Logistic Regression) to predict whether a breast tumor is **Benign** (non-cancerous) or **Malignant** (cancerous) based on cell nucleus measurements.
It is trained on the Wisconsin Breast Cancer Dataset and demonstrates data preprocessing (StandardScaler), model training, evaluation, and an interactive Streamlit web app.

---

## 📘 Project Overview

The main goal of this project is to build a predictive system that can classify tumors as **Benign** or **Malignant** using input medical parameters such as:

- Mean Radius
- Mean Texture
- Mean Perimeter
- Mean Area
- Mean Smoothness
- Mean Compactness
- Mean Concavity
- Mean Concave Points
- Mean Symmetry
- Mean Fractal Dimension
- *(And 20 more "worst" and "standard error" features)*

The project includes:

- Data loading and exploration
- Data standardization using `StandardScaler` (essential for Logistic Regression)
- Model training using Logistic Regression
- Evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Visualization: Confusion Matrix and ROC Curve
- An interactive Streamlit web app for real-time predictions

---

## 🧠 Technologies Used

- Python 🐍
- pandas
- NumPy
- scikit-learn (LogisticRegression, StandardScaler, train_test_split, metrics)
- matplotlib & seaborn (visualizations)
- joblib (model saving)
- streamlit (web application)

---

## 📂 Dataset

**Wisconsin Breast Cancer Dataset** (built into `sklearn.datasets`)

The dataset contains 569 samples with 30 features of cell nuclei and a target label indicating whether the tumor is malignant (0) or benign (1).

🔗 **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/Breast_Cancer_Classification_Logistic
