# Regular Packages
import logging
from typing import Tuple, Dict

# Data Science Packages
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# ML Packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set up logging and seed
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
seed = 1221
np.random.seed(seed)


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a csv file.

    Parameters
    ----------
    path : str
        Path to the csv file.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of the csv file.
    """
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    return df


def undersample_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Undersample the majority class to balance the data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe

    Returns
    -------
    df_downsampled : pd.DataFrame
        Undersampled dataframe.
    """
    logger.info(f"Undersampling data")
    X = df["text"]
    y = df["category"]
    # y = df['categoryId']

    minority_class = y.value_counts().idxmin()
    df_minority = df[df["category"] == minority_class]
    df_downsampled = df_minority

    for category in y.unique():
        if category != minority_class:
            df_majority = df[df["category"] == category]
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=df_minority.shape[0],
                random_state=seed,
            )
            df_downsampled = pd.concat([df_downsampled, df_majority_downsampled])

    return df_downsampled


def smote_data(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Oversample the minority class to balance the data using SMOTE.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.DataFrame
        Target.

    Returns
    -------
    smote_X : pd.DataFrame
        Oversampled features.
    smote_y : pd.DataFrame
        Oversampled target.
    """
    logger.info(f"Oversampling data using SMOTE")
    smote = SMOTE(random_state=seed)
    smote_X, smote_y = smote.fit_resample(X, y)  # type: ignore
    return smote_X, smote_y  # type: ignore


def split_data(
    df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    test_size : float
        Test size.

    Returns
    -------
    X_train : pd.DataFrame
        Train features.
    X_test : pd.DataFrame
        Test features.
    """
    logger.info(
        f"Splitting data into train and test sets with test size of {test_size}"
    )
    X = df["text"]
    y = df["category"]
    # y = df['categoryId']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def tfidf_vectorizer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    ngram_range: tuple,
) -> tuple:
    logger.info(
        f"Vectorizing text data using TF-IDF vectorizer with ngram_range of {ngram_range}"
    )
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test


def process_data(
    path: str,
    test_size: float,
    ngram_range: Tuple = (1, 2),
    data_type: str = "benchmark",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process data for ML models.

    Parameters
    ----------
    path : str
        Path to the csv file.
    test_size : float
        Test size.
    ngram_range : Tuple, optional
        Ngram range for TF-IDF vectorizer, by default (1, 2)
    data_type : str, optional
        Type of data to process, by default 'benchmark'

    Returns
    -------
    X_train : pd.DataFrame
        Train features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.DataFrame
        Train target.
    y_test : pd.DataFrame
        Test target.
    """
    df = load_data(path)
    if data_type == "undersample":
        df = undersample_data(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)
    X_train, X_test = tfidf_vectorizer(X_train, X_test, ngram_range=ngram_range)
    if data_type == "smote":
        X_train, y_train = smote_data(X_train, y_train)
    else:
        pass

    logger.info(f"Train set has {X_train.shape[0]} samples")
    logger.info(f"Test set has {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def sequence_data(
    path: str, test_size: float, data_type: str = "benchmark", max_len: int = 70
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Process data for LSTM models.

    Parameters
    ----------
    path : str
        Path to the csv file.
    test_size : float
        Test size.
    data_type : str, optional
        Type of data to process, by default 'benchmark'
    max_len : int, optional
        Max length of the sequence, by default 70

    Returns
    -------
    X_train : pd.DataFrame
        Train features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.DataFrame
        Train target.
    y_test : pd.DataFrame
        Test target.
    """
    df = load_data(path)
    if data_type == "undersample":
        df = undersample_data(df)

    X = df["text"]
    # y = df['category']
    # make a dictionary of category to integer
    labels = (
        df[["category", "categoryId"]]
        .drop_duplicates()
        .sort_values("categoryId")
        .reset_index(drop=True)
    )
    labels = dict(zip(labels["category"], labels["categoryId"]))
    y = df["categoryId"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Tokenize data
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train.values)
    X_train = tokenizer.texts_to_sequences(X_train.values)
    X_test = tokenizer.texts_to_sequences(X_test.values)

    # Pad data
    max_len = max_len
    X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
    X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

    if data_type == "smote":
        X_train, y_train = smote_data(X_train, y_train)

    return X_train, X_test, y_train, y_test, labels


if __name__ == "__main__":
    path = "../data/clean_data.csv"
    df = load_data(path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    X_train, X_test = tfidf_vectorizer(X_train, X_test, ngram_range=(1, 1))
    X_train, y_train = smote_data(X_train, y_train)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
