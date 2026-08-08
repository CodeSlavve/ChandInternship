import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title='House Price Prediction', layout='centered')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'housing.csv')

# Load model
if not os.path.exists(MODEL_PATH):
    st.error(f'Model file not found at: {MODEL_PATH}')
    st.stop()

with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)
    if isinstance(saved, dict) and 'pipeline' in saved:
        pipeline = saved['pipeline']
    else:
        pipeline = saved

# Helper: extract expected columns from pipeline (ColumnTransformer)
def get_expected_columns(pipe):
    try:
        pre = None
        if hasattr(pipe, 'named_steps'):
            pre = pipe.named_steps.get('preprocessor')
            if pre is None:
                for name, step in pipe.named_steps.items():
                    if hasattr(step, 'transformers_'):
                        pre = step
                        break
        if pre is None:
            return []
        cols = []
        for name, trans, columns in getattr(pre, 'transformers_', []):
            if isinstance(columns, (list, tuple)):
                cols.extend(list(columns))
            elif hasattr(columns, 'tolist'):
                try:
                    cols.extend(list(columns.tolist()))
                except Exception:
                    pass
        seen = set(); out = []
        for c in cols:
            if c not in seen:
                seen.add(c); out.append(c)
        return out
    except Exception:
        return []

expected_cols = get_expected_columns(pipeline)

# Top 15 features computed from dataset
TOP_FEATURES = [
    'Overall Qual',
    'Gr Liv Area',
    'Garage Cars',
    'Garage Area',
    'Total Bsmt SF',
    '1st Flr SF',
    'Year Built',
    'Full Bath',
    'Year Remod/Add',
    'Garage Yr Blt',
    'Mas Vnr Area',
    'TotRms AbvGrd',
    'Fireplaces',
    'BsmtFin SF 1',
    'Lot Frontage'
]

# Try to read data to get sensible ranges; if not available, use defaults
data = None
if os.path.exists(DATA_PATH):
    try:
        data = pd.read_csv(DATA_PATH)
    except Exception:
        data = None

st.title('House Price Prediction')
st.write('Provide values for the top 15 features and click Predict')

# Friendly labels for display
LABELS = {
    'Overall Qual': 'Overall Quality (1-10)',
    'Gr Liv Area': 'Above-grade Living Area (sq ft)',
    'Garage Cars': 'Garage Capacity (number of cars)',
    'Garage Area': 'Garage Area (sq ft)',
    'Total Bsmt SF': 'Total Basement Area (sq ft)',
    '1st Flr SF': 'First Floor Area (sq ft)',
    'Year Built': 'Year Built',
    'Full Bath': 'Number of Full Bathrooms',
    'Year Remod/Add': 'Year Remodeled/Added',
    'Garage Yr Blt': 'Garage Year Built',
    'Mas Vnr Area': 'Masonry Veneer Area (sq ft)',
    'TotRms AbvGrd': 'Total Rooms Above Grade',
    'Fireplaces': 'Number of Fireplaces',
    'BsmtFin SF 1': 'Basement Finished SF Type 1',
    'Lot Frontage': 'Lot Frontage (linear feet)'
}

input_vals = {}
for feat in TOP_FEATURES:
    display_label = LABELS.get(feat, feat)
    # if data available and feature exists, use its distribution to set widget ranges
    if data is not None and feat in data.columns:
        series = data[feat].dropna()
        if pd.api.types.is_numeric_dtype(series):
            lo = float(series.min())
            hi = float(series.max())
            med = float(series.median())
            # For specific features prefer number_input over slider
            prefer_number_input = feat in [
                'Gr Liv Area','Garage Area','Total Bsmt SF','1st Flr SF',
                'Year Built','Year Remod/Add','Garage Yr Blt','Mas Vnr Area',
                'BsmtFin SF 1','Lot Frontage'
            ]
            if prefer_number_input:
                # Use integer input if dtype integer
                if pd.api.types.is_integer_dtype(series):
                    input_vals[feat] = st.number_input(display_label, int(lo), int(hi), int(med), step=1, format='%d')
                else:
                    input_vals[feat] = st.number_input(display_label, float(lo), float(hi), float(med), format='%.2f')
            else:
                # choose integer slider for integer-like columns
                if pd.api.types.is_integer_dtype(series):
                    try:
                        input_vals[feat] = st.slider(display_label, int(lo), int(hi), int(med), step=1)
                    except Exception:
                        input_vals[feat] = st.number_input(display_label, value=med)
                else:
                    # if range is large, use number_input, else slider
                    if hi - lo <= 10000:
                        step = max((hi - lo) / 100.0, 1e-3)
                        input_vals[feat] = st.slider(display_label, float(lo), float(hi), float(med), step=step)
                    else:
                        input_vals[feat] = st.number_input(display_label, value=med, format='%.2f')
        else:
            # categorical-like: offer selectbox of top values
            opts = sorted(series.unique().tolist())
            if len(opts) <= 30:
                input_vals[feat] = st.selectbox(display_label, opts)
            else:
                input_vals[feat] = st.text_input(display_label, '')
    else:
        # fallback sensible defaults per feature name
        if feat in ['Overall Qual']:
            input_vals[feat] = st.slider(display_label, 1, 10, 6)
        elif feat in ['Year Built', 'Year Remod/Add', 'Garage Yr Blt']:
            input_vals[feat] = st.number_input(display_label, min_value=1800, max_value=2100, value=2000)
        elif feat in ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area', 'Mas Vnr Area', 'BsmtFin SF 1', 'Lot Frontage']:
            input_vals[feat] = st.number_input(display_label, value=500.0, format='%.2f')
        elif feat in ['Garage Cars', 'Full Bath', 'TotRms AbvGrd', 'Fireplaces', 'Bedroom AbvGr']:
            input_vals[feat] = st.slider(display_label, 0, 10, 2)
        else:
            input_vals[feat] = st.number_input(display_label, value=0.0)

st.write('')
if st.button('Predict'):
    # Prepare dataframe for prediction
    row = {k: v for k, v in input_vals.items()}
    if expected_cols:
        pdf = pd.DataFrame(columns=expected_cols)
        for c in expected_cols:
            pdf.at[0, c] = row.get(c, np.nan)
    else:
        pdf = pd.DataFrame([row])

    try:
        preds = pipeline.predict(pdf)
        pred = float(preds[0])
        st.success(f'Predicted SalePrice: {pred:,.2f}')
    except Exception as e:
        st.error('Prediction failed')
        st.exception(e)
        st.write('Prepared input:')
        st.write(pdf.head())

st.markdown('---')
st.caption('Top 15 features used as inputs: ' + ', '.join(TOP_FEATURES))