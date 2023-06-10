# Regular Packages
import logging
import re
import yaml
from pathlib import Path

# Data Science Packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Custom Packages
from news.constant import PACKAGE_ROOT_PATH
from news.dataset import undersample_data
from news.utils import confusion_matrix, get_results

# Loading the config file.
config_path = PACKAGE_ROOT_PATH / "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
seed = 1221
np.random.seed(seed)


def cv_main():
    """
    Train and test the baseline models with cross validation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logger.info("Cross validation...")
    ngram_range = tuple(config["ngram_range"])
    data_type = config["data_type"]
    to_save = PACKAGE_ROOT_PATH.parent / "results" / "cross_validation" / data_type
    data_path = PACKAGE_ROOT_PATH.parent / "data" / "clean_data.csv"

    # Load data
    df = pd.read_csv(data_path)
    if data_type == "undersample":
        df = undersample_data(df)
    X = df["text"]
    y = df["category"]
    # y = df['categoryId']
    labels = y.unique()

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
    }
    for name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_scores = []
        test_scores = []
        y_preds = []
        y_true = []
        logger.info(f"Training {name} model")
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pipeline = Pipeline(
                [("tfidf", TfidfVectorizer(ngram_range=ngram_range)), (name, model)]
            )
            pipeline.fit(X_train, y_train)
            train_score = pipeline.score(X_train, y_train)
            train_scores.append(train_score)
            test_score = pipeline.score(X_test, y_test)
            test_scores.append(test_score)
            y_pred = pipeline.predict(X_test)
            y_preds.append(y_pred)
            y_true.append(y_test)

        y_preds = pd.DataFrame(np.concatenate(y_preds))
        y_true = pd.DataFrame(np.concatenate(y_true))
        logger.info(
            f"{name} model has a training score of {np.mean(train_scores):.2f} +/- {np.std(train_scores):.2f}"
        )
        logger.info(
            f"{name} model has a testing score of {np.mean(test_scores):.2f} +/- {np.std(test_scores):.2f}"
        )
        confusion_matrix(y_true, y_preds, name, labels=labels, to_save=to_save)
        get_results(y_true, y_preds, name, labels=labels, to_save=to_save)
    return


if __name__ == "__main__":
    cv_main()
