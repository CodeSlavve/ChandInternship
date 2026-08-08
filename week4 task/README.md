# House Price Prediction (Week 4 Task)

A small project to predict house sale prices using the provided housing dataset. The project includes EDA, preprocessing, training multiple regression models, model selection, saving the best model, PCA exploration, and a simple Streamlit app for inference.

## Dataset

`data/housing.csv` — tabular housing dataset with features such as living area, basement area, garage details, year built, neighborhood, and the target column `SalePrice`.

## Project Structure

```
.
├── app.py                         # Streamlit app for inference (uses model/model.pkl)
├── data/
│   └── housing.csv                # Dataset
├── model/
│   └── model.pkl                  # Pickled best pipeline (preprocessor + model)
├── notebooks/
│   └── house-price-prediction.ipynb # Main Notebook
├── requirements.txt               # Python dependencies
└── README.md
```

## Setup Instructions

1. **Open the project**

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # or: source venv/bin/activate  # macOS / Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the notebooks

- Open the main notebook to reproduce preprocessing, training, and evaluation:
  ```bash
  jupyter notebook notebooks/house-price-prediction.ipynb
  ```
  - Run cells in order (Kernel → Restart & Run All) to reproduce preprocessing, model training, evaluation, and saving the best model into `model/model.pkl`.



## Running the Streamlit app (inference)

The app reads `model/model.pkl` (a pickled dict containing a `pipeline` or a pipeline object) and presents a simple form for the top 15 features.

There is no need to run notebook if you just want to evaluate model.

1. Start the app:
   ```bash
   streamlit run app.py
   ```
2. Interact with the input controls and click **Predict** to see the predicted SalePrice.





## Outputs

-
- `model/model.pkl` — serialized pipeline saved by the notebook. Load with Python's `pickle`.

## Requirements

See `requirements.txt`. Core libraries include: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `xgboost`, `matplotlib`, and `seaborn`.

## Notes

- The notebook selects the best model based on test RMSE (lower is better) and saves the pipeline (preprocessor + trained model) to `model/model.pkl`.

- To change which features the Streamlit app shows, edit `TOP_FEATURES` inside `app.py`.


