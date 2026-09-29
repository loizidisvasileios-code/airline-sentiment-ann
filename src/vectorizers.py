"""Label encoding, dataset splitting and feature extraction."""

import numpy as np
from gensim.models import Word2Vec  # Learns word embeddings from our own corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_SEED, LABEL_COL


def encode_labels(df):
    """Turn the three sentiment names into the integers CrossEntropyLoss expects."""
    encoder = LabelEncoder()  # Maps sorted unique strings to 0, 1, 2
    df["label"] = encoder.fit_transform(df[LABEL_COL])
    class_names = list(encoder.classes_)  # Kept so plots can show names instead of numbers
    print("Label mapping:", {name: index for index, name in enumerate(class_names)})
    return df, class_names


def split_data(df):
    """Split 80/10/10 into train, validation and test, preserving the class balance."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,  # 20% is held out, to be halved below
        random_state=RANDOM_SEED,  # Same split on every run
        stratify=df["label"],  # Each part keeps the 62.7 / 21.2 / 16.1 proportions
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # Half of the 20%, so 10% of the full dataset each
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )
    print(f"Train {len(train_df)} | Validation {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


def build_tfidf_features(train_texts, val_texts, test_texts, max_features=5000):
    """Fit TF-IDF on the training texts only, then apply the fitted vectorizer to all splits.

    Fitting on the training split alone is what keeps information about the test set out
    of the IDF weights, which is the difference between an honest and an optimistic score.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,  # Keep the most frequent terms, which also limits overfitting
        min_df=2,  # Ignore terms that appear in a single document and cannot generalise
        ngram_range=(1, 2),  # Single words plus word pairs, so "not good" survives as one feature
    )
    X_train = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)  # Learns and transforms
    X_val = vectorizer.transform(val_texts).toarray().astype(np.float32)  # Transforms only
    X_test = vectorizer.transform(test_texts).toarray().astype(np.float32)  # Transforms only
    print(f"TF-IDF: {X_train.shape[1]} features | training matrix {X_train.nbytes / 1e6:.0f} MB")
    return X_train, X_val, X_test, vectorizer


def _average_vectors(token_lists, model, fallback):
    """Turn each tweet into one vector by averaging the vectors of the words it contains."""
    matrix = []
    for tokens in token_lists:
        vectors = [model.wv[token] for token in tokens if token in model.wv]  # Skip unknown words
        matrix.append(np.mean(vectors, axis=0) if vectors else fallback)  # Fallback prevents NaN
    return np.array(matrix, dtype=np.float32)


def build_word2vec_features(train_tokens, val_tokens, test_tokens, vector_size=100):
    """Train Word2Vec on the training tokens only, then average word vectors per tweet."""
    model = Word2Vec(
        sentences=list(train_tokens),  # Only the training split, for the same reason as TF-IDF
        vector_size=vector_size,  # Dimensions per word vector
        window=5,  # How many neighbouring words count as context
        min_count=2,  # Ignore words seen once, matching the TF-IDF min_df
        sg=1,  # Skip-gram, which the lecture recommends for smaller corpora
        seed=RANDOM_SEED,
        workers=1,  # A single thread is what makes the training reproducible
    )
    fallback = model.wv.vectors.mean(axis=0)  # Average of every learned vector, for empty tweets
    X_train = _average_vectors(train_tokens, model, fallback)
    X_val = _average_vectors(val_tokens, model, fallback)
    X_test = _average_vectors(test_tokens, model, fallback)
    print(f"Word2Vec: {vector_size} dimensions | vocabulary {len(model.wv)} words")
    return X_train, X_val, X_test, model
