"""Process raw dataset files into standardized CSV for training.
Parses common formats (UCI SMSSpamCollection tab-separated, CSVs), normalizes to columns: 'label' and 'text'.
Saves processed CSV to data/processed/processed.csv by default.
"""
import argparse
import os
import pandas as pd
from src.preprocessing import clean_text


def try_read(path):
    # Try common read formats
    # Detect delimiter by inspecting the first line to avoid mis-parsing CSV as tab-separated
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first = f.readline()
    except Exception:
        first = ''

    # 1) If file looks tab-separated (UCI format), read with tab
    if '\t' in first:
        try:
            df = pd.read_csv(path, sep='\t', header=None, names=['label', 'text'], encoding='latin-1')
            if df.shape[1] >= 2:
                return df[['label', 'text']]
        except Exception:
            pass

    # 2) Comma-separated with header
    try:
        df = pd.read_csv(path, encoding='utf-8')
        # heuristic to find label/text columns (handle emails.csv with columns: text,spam)
        cols = [c.lower() for c in df.columns]
        # if explicit columns present
        if ('label' in cols or 'spam' in cols) and ('text' in cols or 'message' in cols):
            # find text column
            text_col = None
            for candidate in ['text', 'message', 'body', 'subject']:
                if candidate in cols:
                    text_col = df.columns[cols.index(candidate)]; break
            # find label column
            label_col = None
            for candidate in ['label', 'spam', 'is_spam']:
                if candidate in cols:
                    label_col = df.columns[cols.index(candidate)]; break
            if text_col and label_col:
                return df[[label_col, text_col]].rename(columns={label_col: 'label', text_col: 'text'})
        # if only text and a numeric label column at second position (like emails.csv: text,spam)
        if df.shape[1] >= 2:
            # prefer if first column looks like text (long strings)
            return df.iloc[:, :2].rename(columns={df.columns[0]: 'text', df.columns[1]: 'label'})
    except Exception:
        pass
    # 3) fallback: read as table splitting by first whitespace/tab
    with open(path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        line = line.strip('\n')
        if not line:
            continue
        # split at first tab or first whitespace sequence
        if '\t' in line:
            lab, msg = line.split('\t', 1)
        else:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                lab, msg = parts
            else:
                continue
        rows.append({'label': lab, 'text': msg})
    return pd.DataFrame(rows)


def normalize_labels(df):
    # Normalize label column to integers 0 (ham/legit) and 1 (spam)
    df = df.copy()
    if 'label' not in df.columns:
        return df
    # If label is textual (e.g., 'ham'/'spam')
    if df['label'].dtype == object:
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        mapping = {'ham': 0, 'spam': 1}
        if set(df['label'].unique()).issubset(set(mapping.keys())):
            df['label'] = df['label'].map(mapping)
    # If label is numeric-like but not 0/1, try to coerce
    try:
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        # If values are like 1/2 or 1/0; map >0 to 1
        if set(df['label'].dropna().unique()) - {0, 1}:
            df['label'] = (df['label'] > 0).astype(int)
        else:
            df['label'] = df['label'].fillna(0).astype(int)
    except Exception:
        pass
    return df


def preprocess_and_save(input_path, output_path):
    df = try_read(input_path)
    if df is None or df.empty:
        raise RuntimeError('Unable to parse input dataset: ' + input_path)
    # Ensure columns: final DataFrame must have 'text' and 'label'
    cols = [c.lower() for c in df.columns]
    if not (('label' in cols) and ('text' in cols)):
        # if first column looks like text, map first->text second->label
        first = cols[0]
        if first in ('text', 'message', 'body', 'subject'):
            df = df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'label'})
        else:
            # default fallback: assume first is label, second is text
            df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'text'})
    # Drop NA text
    df = df.dropna(subset=['text'])
    # Clean text
    df['text'] = df['text'].astype(str).apply(clean_text)
    # Normalize labels
    df = normalize_labels(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'Processed {len(df)} rows and wrote to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Raw dataset file path')
    parser.add_argument('--output', default='data/processed/processed.csv', help='Output processed CSV path')
    args = parser.parse_args()
    preprocess_and_save(args.input, args.output)
