# Regular Packages
import logging
import re
import yaml
from pathlib import Path


# Data Science Packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Custom Packages
from news.constant import PACKAGE_ROOT_PATH
from news.dataset import process_data
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


def benchmark_main():
    """
    Train and test the baseline models.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    test_size = config["test_size"]
    ngram_range = tuple(config["ngram_range"])
    data_type = config["data_type"]
    weighted_loss = config["weighted_loss"]
    to_save = PACKAGE_ROOT_PATH.parent / "results" / "base_line" / data_type
    data_path = PACKAGE_ROOT_PATH.parent / "data" / "clean_data.csv"
    X_train, X_test, y_train, y_test = process_data(data_path, test_size=test_size, ngram_range=ngram_range, data_type=data_type)  # type: ignore
    labels = y_test.unique()  # type: ignore

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
    }
    if weighted_loss:
        models["Logistic Regression"] = LogisticRegression(class_weight="balanced")
        models["Random Forest"] = RandomForestClassifier(class_weight="balanced")
        models["SVM"] = SVC(probability=True, class_weight="balanced")
        to_save = to_save / "weighted_loss"

    for name, model in models.items():
        logger.info(f"Training {name} model")
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        logger.info(f"{name} model has a training score of {train_score:.2f}")
        logger.info(f"Testing {name} model")
        test_score = model.score(X_test, y_test)
        logger.info(f"{name} model has a testing score of {test_score:.2f}")

        y_pred = model.predict(X_test)
        confusion_matrix(y_test, y_pred, name, labels=labels, to_save=to_save)
        get_results(y_test, y_pred, name, labels=labels, to_save=to_save)
    return


if __name__ == "__main__":
    benchmark_main()
