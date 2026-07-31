"""Feature extraction utilities
- simple Bag-of-Words / TF-IDF wrappers
- saving and loading vectorizers
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def build_vectorizer(corpus, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features)
    vec.fit(corpus)
    return vec

def save_vectorizer(vec, path):
    joblib.dump(vec, path)

def load_vectorizer(path):
    return joblib.load(path)
