Diabetes Prediction using SVM

This project uses Machine Learning (Support Vector Machine) to predict whether a person is diabetic or non-diabetic based on their medical details.
It is trained on the PIMA Diabetes Dataset and demonstrates data preprocessing, model training, and prediction steps.

📘 Project Overview

The main goal of this project is to build a predictive system that can classify individuals as diabetic or non-diabetic using input medical parameters such as:

Number of pregnancies

Glucose level

Blood pressure

Skin thickness

Insulin level

BMI

Diabetes pedigree function

Age

The project includes:

Data loading and exploration

Data standardization

Model training using Support Vector Machine (SVM)

Accuracy evaluation

Prediction on new input data

🧠 Technologies Used

Python 🐍

pandas

NumPy

scikit-learn (for SVM, preprocessing, model evaluation)

📂 Dataset

PIMA Diabetes Dataset

The dataset contains information on medical details of patients and their diabetes status.
You can download it from Kaggle:
🔗 PIMA Diabetes Dataset on Kaggle

⚙️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/huzaifaYounas443/Diabetes-Prediction.git
cd Diabetes-Prediction

2️⃣ Install Required Packages
pip install numpy pandas scikit-learn

3️⃣ Run the Python File
python diabetes_prediction.py

🧩 Project Structure
Diabetes-Prediction/
│
├── diabetes.csv                # Dataset file
├── diabetes_prediction.py      # Main Python script
├── README.md                   # Project documentation
└── requirements.txt            # (optional) List of dependencies

📊 Model Details
Algorithm Used: Support Vector Machine (SVM)

Kernel: Linear
Train-Test Split: 80% training, 20% testing

Performance:

Training Accuracy ≈ 78–80%
Testing Accuracy ≈ 75–78%

🧪 Example Prediction

Input:
(4, 110, 92, 0, 0, 37.6, 0.191, 30)


Output:
The person is not diabetic
