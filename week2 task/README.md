# Loan Approval Prediction

Predicts whether a loan application will be **Approved** or **Rejected** using applicant financial and demographic data, comparing multiple ML models with hyperparameter tuning.

## Dataset

`loan_approval_dataset.csv` — contains applicant details: dependents, education, employment status, income, loan amount, loan term, CIBIL score, and asset values (residential, commercial, luxury, bank), with the target column `loan_status`.

## Project Structure

```
.
├── loan-approval-prediction.ipynb   # Main notebook: EDA, preprocessing, modeling, tuning, visual reports
├── loan_approval_dataset.csv        # Dataset
├── requirements.txt                 # Python dependencies
└── README.md
```

## Setup Instructions

1. **Clone / extract the project**
   ```bash
   unzip loan-approval-prediction.zip
   cd loan-approval-prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook loan-approval-prediction.ipynb
   ```
   Run all cells top to bottom (**Kernel → Restart & Run All**) to reproduce EDA, model training, tuning, and visual reports.

## What the Notebook Does

1. **EDA** — null checks, duplicates, distribution plots (income, loan amount), boxplots, correlation heatmap, pairplot.
2. **Preprocessing** — drops `loan_id`, label-encodes categorical columns (`education`, `self_employed`, `loan_status`).
3. **Baseline Modeling** — trains and compares Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost on Accuracy, Precision, Recall, F1, and ROC AUC.
4. **Hyperparameter Tuning** — `GridSearchCV` (5-fold) on Random Forest and XGBoost.
5. **Visual Reports** — baseline model comparison chart, CV vs test accuracy chart, confusion matrices, ROC curves, and feature importance for the best tuned model.

## Requirements

See `requirements.txt`. Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`.

## Notes

- Best model is selected automatically based on Test Accuracy/ROC AUC in the tuning results table.