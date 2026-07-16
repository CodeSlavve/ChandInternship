# Student Performance Prediction

## Machine Learning Internship - Week 1

A machine learning project that predicts students' final academic performance using the UCI Student Performance dataset. The project demonstrates the complete ML workflow including data preprocessing, feature selection, model training, evaluation, and visualization using Python and Scikit-learn.

---

## Project Objective

The objective of this project is to predict a student's final grade (G3) based on demographic, academic, and social factors. Two regression models are implemented and compared:

- Linear Regression
- Random Forest Regressor

---

## Dataset

**Source:** [UC Irvine's Student Performace dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

Dataset: Student Performance Dataset

File Used:
- `student-mat.csv`

The dataset contains student information such as:

- Age
- Study time
- Previous grades
- Family background
- Absences
- Health
- Internet access
- Extra educational support
- And several other academic/social attributes

Target Variable:

- **G3** (Final Grade)

---

## Project Workflow

### 1. Data Loading

- Load dataset using Pandas
- Explore dataset
- Display summary statistics
- Check data types
- Check for missing values

---

### 2. Data Preprocessing

- Label Encoding for categorical features
- Correlation matrix generation
- Data scaling using StandardScaler

---

### 3. Feature Selection

For Linear Regression:

- Pearson Correlation-based feature selection
- Top 10 most correlated features selected

For Random Forest:

- Embedded feature selection using Feature Importance

---

### 4. Model Training

Two machine learning models are trained:

#### Linear Regression

- Standardized input features
- Correlation-selected features

#### Random Forest Regressor

- Trained on complete feature set
- Feature importance extracted automatically

---

### 5. Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Actual and predicted values are also compared.

---

### 6. Visualization

The notebook includes several visualizations:

- Correlation Heatmap
- Actual vs Predicted Plot
- Random Forest Feature Importance
- Distribution of Final Grades
- Model Performance Comparison

---

## Technologies Used

- Python 3.13.1
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure

```
Student_Performance_Prediction/
│
├── student_score_prediction.ipynb
├── student-mat.csv
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository or extract the project ZIP.

Install the required libraries:

```bash
pip install -r requirements.txt
```

or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
```

---

## Running the Project

### Option 1: Using Visual Studio Code

1. Open the project folder in **Visual Studio Code**.
2. Install the **Python** and **Jupyter** extensions if they are not already installed.
3. Create and activate a virtual environment (optional).

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Open `student_score_prediction.ipynb`.

6. Select the Python interpreter (or the `.venv` environment) from the top-right corner.

7. Click **Run All** to execute all notebook cells.

---

### Option 2: Using Jupyter Notebook

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
student_score_prediction.ipynb
```

Run all cells sequentially.

---

## Machine Learning Models

| Model | Purpose |
|--------|----------|
| Linear Regression | Baseline regression model |
| Random Forest Regressor | Ensemble learning model with embedded feature selection |

---

## Evaluation Metrics

The following metrics are used to compare model performance:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Lower MAE and RMSE indicate better prediction accuracy, while a higher R² score indicates a better fit.

---

## Features Implemented

- Data preprocessing
- Label encoding
- Correlation analysis
- Feature selection
- Feature scaling
- Model training
- Prediction
- Model evaluation
- Data visualization
- Feature importance analysis