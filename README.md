# 🩺 Diabetes Prediction using Machine Learning

A simple machine learning project that predicts whether a patient is likely to have diabetes based on medical attributes.

## 📊 Dataset

This project uses the **Pima Indians Diabetes Dataset** — a well-known dataset in the ML community.

**Features used:**
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |

**Target:** `Outcome` — 0 (No Diabetes) or 1 (Diabetes)

## 🧠 Model

- **Algorithm:** Logistic Regression
- **Preprocessing:** StandardScaler for feature scaling
- **Train/Test Split:** 80/20

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd diabetic-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook diabetes_prediction.ipynb
   ```

4. Run all cells to train the model and see predictions on new patient data.

## 📁 Project Structure

```
├── diabetes_prediction.ipynb   # Main notebook (train + predict)
├── training_data.csv           # Training dataset (768 samples)
├── new_patients.csv            # New patient data for prediction
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

## 🛠️ Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook
