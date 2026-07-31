"""Training script (CLI-friendly)
Example usage:
python train.py --data data/processed/train.csv --model models/spam_nb.joblib
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
from src.features import build_vectorizer
from src.preprocessing import clean_text


def main(data_path, model_out):
    df = pd.read_csv(data_path)
    # Detect label column ('label' or 'spam')
    if 'label' in df.columns:
        label_col = 'label'
    elif 'spam' in df.columns:
        label_col = 'spam'
    else:
        # fallback to second column
        label_col = df.columns[1]

    # Ensure text is string and fill missing
    if 'text' not in df.columns:
        # try to find a likely text column
        for c in df.columns:
            if c.lower() in ('text', 'message', 'body', 'subject'):
                df = df.rename(columns={c: 'text'})
                break
    df['text'] = df['text'].fillna('').astype(str)
    df['clean_text'] = df['text'].apply(lambda x: clean_text(x))

    # Build vectorizer and transform
    vec = build_vectorizer(df['clean_text'])
    X = vec.transform(df['clean_text'])

    # Prepare labels
    y = df[label_col]
    y = y.fillna(0)
    # If labels are strings like 'ham'/'spam' converted earlier, ensure numeric
    if y.dtype == object:
        y = y.astype(str).str.strip().map({'ham': 0, 'spam': 1}).fillna(y)
    try:
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
    except Exception:
        pass

    # Train/test split with stratify if possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump({'model': model, 'vec': vec}, model_out)
    print('Saved model to', model_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', required=True)
    parser.add_argument('--model', dest='model', required=True)
    args = parser.parse_args()
    main(args.data, args.model)
