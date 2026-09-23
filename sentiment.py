"""CSCI 544 HW1, Fall 2026. Complete and submit this file as sentiment.py.

Implement the documented functions; you may add helpers and imports.
Use the handout and requirements.txt. Do not download resources at runtime.
"""

import csv
import html
import math
import re
from collections import Counter
from functools import lru_cache

import contractions
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

RANDOM_STATE = 42
MODEL_NAMES = ["Perceptron", "Logistic Regression", "Naive Bayes"]

_TAG_RE = re.compile(r"<[^>]*>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WS_RE = re.compile(r"\s+")

UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"


# ---------------------------------------------------------------------------
# Section 1: dataset preparation
# ---------------------------------------------------------------------------

def prepare_data(path, n_per_class=100000):
    """Return (train_df, test_df, stats), as specified in Section 1/interface."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        on_bad_lines="skip",
        usecols=["review_body", "star_rating"],
    )
    df = df.rename(columns={"review_body": "review", "star_rating": "rating"})
    df = df[["review", "rating"]]
    n_loaded = len(df)

    ratings = pd.to_numeric(df["rating"], errors="coerce")
    review_ok = df["review"].notna() & (df["review"].str.strip() != "")
    rating_ok = ratings.isin([1, 2, 3, 4, 5])
    valid = review_ok & rating_ok

    data = pd.DataFrame({
        "review": df.loc[valid, "review"],
        "rating": ratings[valid].astype(int),
    })
    excluded_rows = int(n_loaded - len(data))

    counts = data["rating"].value_counts()
    rating_counts = {r: int(counts.get(r, 0)) for r in range(1, 6)}
    class_counts = {
        "positive": rating_counts[4] + rating_counts[5],
        "negative": rating_counts[1] + rating_counts[2],
        "neutral": rating_counts[3],
    }

    data = data[data["rating"] != 3].copy()
    data["label"] = (data["rating"] >= 4).astype(int)

    pos = data[data["label"] == 1].sample(
        n=n_per_class, replace=False, random_state=RANDOM_STATE)
    neg = data[data["label"] == 0].sample(
        n=n_per_class, replace=False, random_state=RANDOM_STATE)
    sample = pd.concat([pos, neg]).reset_index(drop=True)
    sample = sample[["review", "rating", "label"]]

    train_df, test_df = train_test_split(
        sample,
        test_size=0.2,
        stratify=sample["label"],
        random_state=RANDOM_STATE,
    )

    stats = {
        "rating_counts": rating_counts,
        "class_counts": class_counts,
        "excluded_rows": excluded_rows,
    }
    return train_df, test_df, stats


# ---------------------------------------------------------------------------
# Sections 2 and 3: cleaning and preprocessing
# ---------------------------------------------------------------------------

def clean_review(text):
    """Return one cleaned string, including the empty string when applicable."""
    text = html.unescape(text).lower()
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = contractions.fix(text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=1)
def _stop_words():
    return frozenset(stopwords.words("english")) - {"no", "nor", "not"}


@lru_cache(maxsize=1)
def _lemmatizer():
    return WordNetLemmatizer()


@lru_cache(maxsize=None)
def _lemmatize(token):
    return _lemmatizer().lemmatize(token)


def preprocess_review(text):
    """Return stopword-filtered, noun-lemmatized text from cleaned input."""
    stops = _stop_words()
    return " ".join(_lemmatize(tok) for tok in text.split() if tok not in stops)


# ---------------------------------------------------------------------------
# Section 4: feature extraction
# ---------------------------------------------------------------------------

def extract_features(train_texts, test_texts):
    """Return (fitted_vectorizer, sparse_X_train, sparse_X_test)."""
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(list(train_texts))
    X_test = vectorizer.transform(list(test_texts))
    return vectorizer, X_train, X_test


# ---------------------------------------------------------------------------
# Sections 5-7: classifiers
# ---------------------------------------------------------------------------

def train_model(name, X_train, y_train):
    """Return a fitted classifier for one of the three specified model names."""
    if name == "Perceptron":
        model = Perceptron(random_state=RANDOM_STATE, max_iter=1000, tol=1e-3)
    elif name == "Logistic Regression":
        model = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs",
                                   max_iter=1000, tol=1e-4)
    elif name == "Naive Bayes":
        model = MultinomialNB(alpha=1.0)
    else:
        raise ValueError(f"Unknown model name: {name}")
    model.fit(X_train, np.asarray(y_train))
    return model


def evaluate_model(model, X, y):
    """Return unrounded accuracy, precision, recall, f1 in a dictionary."""
    y = np.asarray(y)
    y_pred = model.predict(X)
    kwargs = {"average": "binary", "pos_label": 1, "zero_division": 0}
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, **kwargs)),
        "recall": float(recall_score(y, y_pred, **kwargs)),
        "f1": float(f1_score(y, y_pred, **kwargs)),
    }


# ---------------------------------------------------------------------------
# Section 8: bigram language model
# ---------------------------------------------------------------------------

def _map_sequence(text, vocab):
    """Return [<s>, mapped tokens..., </s>] for one cleaned review."""
    tokens = [tok if tok in vocab else UNK for tok in text.split()]
    return [BOS] + tokens + [EOS]


def fit_bigram(train_texts, min_count=3):
    """Return a dictionary with vocab, bigrams, contexts; omit zero counts."""
    word_counts = Counter()
    for text in train_texts:
        word_counts.update(text.split())
    vocab = {w for w, c in word_counts.items() if c >= min_count}
    vocab.update({UNK, EOS})

    bigrams = Counter()
    for text in train_texts:
        seq = _map_sequence(text, vocab)
        bigrams.update(zip(seq[:-1], seq[1:]))

    contexts = Counter()
    for (h, _w), c in bigrams.items():
        contexts[h] += c

    return {"vocab": vocab, "bigrams": dict(bigrams),
            "contexts": dict(contexts)}


def evaluate_bigram(model, test_texts, k=1.0):
    """Return a dictionary with unseen, n_tokens, add-k perplexity; do not refit."""
    vocab = model["vocab"]
    bigrams = model["bigrams"]
    contexts = model["contexts"]
    kv = k * len(vocab)

    unseen = 0
    n_tokens = 0
    log_prob = 0.0
    for text in test_texts:
        seq = _map_sequence(text, vocab)
        for h, w in zip(seq[:-1], seq[1:]):
            c_hw = bigrams.get((h, w), 0)
            if c_hw == 0:
                unseen += 1
            log_prob += math.log((c_hw + k) / (contexts.get(h, 0) + kv))
            n_tokens += 1

    perplexity = math.exp(-log_prob / n_tokens)
    return {"unseen": int(unseen), "n_tokens": int(n_tokens),
            "perplexity": float(perplexity)}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _mean_len(texts):
    return float(np.mean([len(t) for t in texts]))


def run_experiment(path="data.tsv", n_per_class=100000,
                   lm_train_size=5000, lm_test_size=1000):
    """Run the pipeline, print the 37 required lines, and return None."""
    train_df, test_df, stats = prepare_data(path, n_per_class)

    train_raw = train_df["review"].tolist()
    test_raw = test_df["review"].tolist()
    train_clean = [clean_review(t) for t in train_raw]
    test_clean = [clean_review(t) for t in test_raw]
    train_prep = [preprocess_review(t) for t in train_clean]
    test_prep = [preprocess_review(t) for t in test_clean]

    _vectorizer, X_train, X_test = extract_features(train_prep, test_prep)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    cc = stats["class_counts"]
    print(f"Positive reviews: {cc['positive']}")
    print(f"Negative reviews: {cc['negative']}")
    print(f"Neutral reviews: {cc['neutral']}")
    print(f"Average length before cleaning: {_mean_len(train_raw):.4f}")
    print(f"Average length after cleaning: {_mean_len(train_clean):.4f}")
    print(f"Average length before preprocessing: {_mean_len(train_clean):.4f}")
    print(f"Average length after preprocessing: {_mean_len(train_prep):.4f}")
    print(f"TF-IDF matrix shape: ({X_train.shape[0]}, {X_train.shape[1]})")

    metric_labels = [("accuracy", "Accuracy"), ("precision", "Precision"),
                     ("recall", "Recall"), ("f1", "F1-score")]
    for name in MODEL_NAMES:
        model = train_model(name, X_train, y_train)
        for split, X, y in (("Training", X_train, y_train),
                            ("Testing", X_test, y_test)):
            scores = evaluate_model(model, X, y)
            for key, label in metric_labels:
                print(f"{name} {split} {label}: {scores[key]:.4f}")

    # Bigram subsets: cleaned text (before stop-word removal/lemmatization),
    # non-empty only, sampled in split order.
    lm_train_pool = pd.Series(train_clean, index=train_df.index)
    lm_test_pool = pd.Series(test_clean, index=test_df.index)
    lm_train_pool = lm_train_pool[lm_train_pool != ""]
    lm_test_pool = lm_test_pool[lm_test_pool != ""]
    lm_train = lm_train_pool.sample(
        n=lm_train_size, replace=False, random_state=RANDOM_STATE).tolist()
    lm_test = lm_test_pool.sample(
        n=lm_test_size, replace=False, random_state=RANDOM_STATE).tolist()

    lm = fit_bigram(lm_train)
    result_k1 = evaluate_bigram(lm, lm_test, k=1.0)
    result_k01 = evaluate_bigram(lm, lm_test, k=0.1)
    result_k001 = evaluate_bigram(lm, lm_test, k=0.01)

    print(f"Bigram vocabulary size: {len(lm['vocab'])}")
    print(f"Bigram unseen test transitions: {result_k1['unseen']}")
    print(f"Bigram test perplexity: {result_k1['perplexity']:.4f}")
    print(f"Bigram test perplexity (k=0.1): {result_k01['perplexity']:.4f}")
    print(f"Bigram test perplexity (k=0.01): {result_k001['perplexity']:.4f}")
    return None


if __name__ == "__main__":
    run_experiment()
