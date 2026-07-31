Week 3 — Spam Email Detection System

Dataset summary
- Data source: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset
- Why chosen: this email dataset contains real spam and non-spam email text with a binary label. It is well suited for NLP feature extraction and TF-IDF + Naive Bayes modeling, while staying small enough for fast iteration.

Structure
- data/raw/: raw dataset files
- data/processed/: cleaned, normalized CSVs for training
- notebooks/: exploratory notebook such as eda.ipynb
- src/: processing, training, and prediction scripts
- models/: saved trained model artifacts
- requirements.txt: Python dependencies

Quick start
1) Activate Python environment
   - python -m venv .venv
   - .\\.venv\\Scripts\\activate

2) Install dependencies
   - pip install -r requirements.txt

3) Process raw data
   - python -m src.process_data --input data/raw/emails.csv --output data/processed/processed.csv

4) Train model
   - python -m src.train --data data/processed/processed.csv --model models/spam_nb.joblib

5) Test model
   - python -m src.predict --model models/spam_nb.joblib --text "Your sample email text here"
