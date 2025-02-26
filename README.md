# Bank-Customer-Churn-Prediction

## Overview
This project predicts whether a bank customer is likely to churn (leave the bank) based on their demographics, account activity, and other financial indicators. The goal is to help banks improve customer retention strategies by identifying at-risk customers early.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn, XGBoost, TensorFlow/Keras
- **Machine Learning Models:** Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost, Neural Networks

## Dataset
- The dataset (`Churn_Modelling.csv`) contains:
  - Customer demographics (Age, Gender, Geography, etc.)
  - Account details (Balance, Credit Score, Number of Products, etc.)
  - Churn indicator (`Exited` column: 1 for churned, 0 for retained customers)

## Methodology
### 1. Data Preprocessing
- Handled missing values and outliers.
- Encoded categorical variables using **OneHotEncoder** and **LabelEncoder**.
- Standardized numerical features using **StandardScaler**.

### 2. Model Training
- Trained multiple machine learning models, including:
  - **Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost, Neural Networks**
- Used **GridSearchCV** and **Stratified K-Fold Cross-Validation** for hyperparameter tuning.

### 3. Model Evaluation
- Evaluated models using **Accuracy, Precision, Recall, ROC-AUC Score**.
- Plotted **Confusion Matrix** and **ROC Curves** to compare models.

## Results
| Model | Accuracy | ROC-AUC Score |
|--------|----------|--------------|
| Logistic Regression | 80.2% | 0.79 |
| Decision Tree | 82.4% | 0.81 |
| Random Forest | 84.9% | 0.85 |
| SVM (RBF Kernel) | 83.1% | 0.83 |
| SVM (Polynomial Kernel) | 81.6% | 0.81 |
| Stochastic Gradient Descent (SGD) | 79.5% | 0.78 |
| XGBoost | **86.2%** | **0.87** |
| Neural Network | **85.4%** | **0.86** |

- **XGBoost and Neural Networks performed best**, achieving the highest accuracy and recall.
- **Random Forest also performed well** but had slightly lower recall.

## How to Use
```bash
# Clone the repository
git clone https://github.com/yourusername/bank-churn-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the script
python churn_prediction.py
```

## Future Improvements
- Improve feature engineering to enhance model interpretability.
- Implement deep learning models with **LSTMs** for sequential transaction data.
- Develop a **real-time churn prediction API** for banking applications.


