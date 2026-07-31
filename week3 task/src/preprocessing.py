"""Text preprocessing for spam detection
Functions:
- clean_text(text)
- tokenize(text)
- preprocess_dataframe(df, text_col)
"""

import re

# Attempt to use NLTK if available, otherwise fall back to simple split tokenization
try:
    import nltk
    _nltk_available = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # don't fail import if punkt is missing; download on demand later
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
except Exception:
    _nltk_available = False


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    if _nltk_available:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            return text.split()
    else:
        return text.split()

# Add more preprocessing (stopword removal, stemming/lemmatization) as needed
