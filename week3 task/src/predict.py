"""Simple prediction wrapper
Usage:
python predict.py --model models/spam_nb.joblib --text "Free money now"
"""
import argparse
import joblib
from src.preprocessing import clean_text


def predict_text(model_path, text):
    artefact = joblib.load(model_path)
    model = artefact['model']
    vec = artefact['vec']
    clean = clean_text(text)
    X = vec.transform([clean])
    pred = model.predict(X)
    return pred[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    print(predict_text(args.model, args.text))
