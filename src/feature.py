import subprocess
import os
from typing import Tuple, Optional, Union, List

import joblib
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


# Initialize spaCy
def initialize_spacy(model_name: str = "fr_core_news_md") -> spacy.language.Language:
    """Initialize spaCy model"""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        nlp = spacy.load(model_name)
    return nlp


nlp = initialize_spacy()


def process_text_lematization(text: str) -> str:
    """
    Preprocess text using lemmatization and stop words removal

    """
    stop_words = nlp.Defaults.stop_words
    return " ".join(
        [
            token.lemma_.lower()
            for token in nlp(str(text))
            if token.text.lower() not in stop_words and not token.is_punct
        ]
    )


def apply_CountVectorizer(X_train, save_path: str = "models/count_vectorizer.pkl"):
    """
    Apply Count Vectorization to training data

    """
    ensure_dir(save_path)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    joblib.dump(vectorizer, save_path)
    return X_train


def apply_tfidf_vectorizer(X_train, save_path: str = 'models/tfidf_vectorizer.pkl'):
    """
    Apply TF-IDF Vectorization to training data

    """
    ensure_dir(save_path)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    joblib.dump(vectorizer, save_path)
    return X_train


def apply_word2vec(X_train: List[List[str]], save_path: str = "models/word2vec.model") -> np.ndarray:
    """
    Trains a Word2Vec model and transforms training data

    """
    # Train Word2Vec model
    model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)

    # Save model
    ensure_dir(save_path)
    model.save(save_path)

    # Transform sentences to vectors
    X_train_vectors = np.array([
        get_sentence_vector(sentence, model)
        for sentence in X_train
    ])

    return X_train_vectors

def ensure_dir(file_path: str):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)



def load_word2vec_and_transform(X_test: List[List[str]], load_path: str = "models/word2vec.model") -> np.ndarray:
    """
    Loads saved Word2Vec model and transforms test data
    """
    try:
        model = Word2Vec.load(load_path)
        X_test_vectors = np.array([
            get_sentence_vector(sentence, model)
            for sentence in X_test
        ])
        return X_test_vectors
    except FileNotFoundError:
        raise FileNotFoundError(f"Word2Vec model not found at {load_path}")


def get_sentence_vector(sentence: List[str], model: Word2Vec) -> np.ndarray:
    """
    Converts a sentence to its vector representation by averaging word vectors
    """
    word_vectors = [
        model.wv[word]
        for word in sentence
        if word in model.wv
    ]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(model.vector_size)


def load_vectorizer_and_transform(X_test, load_path: str = "models/count_vectorizer.pkl"):
    """Load saved vectorizer and transform test data"""
    vectorizer = joblib.load(load_path)
    return vectorizer.transform(X_test).toarray()


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words

    """
    return text.split()


def make_features(df, train: bool = True, save_path: str = "models/tfidf_vectorizer.pkl",
                  vectorizer_type: str = "tfidf") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create features from DataFrame using specified vectorization method

    """
    # Preprocess text
    df["video_name_lematized"] = df["video_name"].apply(process_text_lematization)

    # Get labels if available
    y = df["is_comic"].values if "is_comic" in df.columns else None

    # Get video titles
    titres_videos = df["video_name_lematized"].fillna("")

    if vectorizer_type.lower() == "word2vec":
        # For Word2Vec, we need tokenized text
        tokenized_texts = [tokenize_text(text) for text in titres_videos]
        if train:
            X = apply_word2vec(tokenized_texts, save_path)
        else:
            X = load_word2vec_and_transform(tokenized_texts, save_path)
    else:
        # For TF-IDF and Count vectorizers
        if train:
            if vectorizer_type.lower() == "tfidf":
                X = apply_tfidf_vectorizer(titres_videos, save_path)
            elif vectorizer_type.lower() == "count":
                X = apply_CountVectorizer(titres_videos, save_path)
            else:
                raise ValueError("vectorizer_type must be one of 'tfidf', 'count', or 'word2vec'")
        else:
            vectorizer = joblib.load(save_path)
            X = vectorizer.transform(titres_videos).toarray()

    return X, y
