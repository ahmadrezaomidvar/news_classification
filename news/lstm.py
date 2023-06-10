# Regular Packages
import logging
import re
import yaml

# Data Science Packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom Packages
from news.constant import PACKAGE_ROOT_PATH
from news.dataset import sequence_data
from news.utils import plot_history, confusion_matrix, get_results

# Loading the config file.
config_path = PACKAGE_ROOT_PATH / "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
seed = 1221
np.random.seed(seed)
tf.random.set_seed(seed)


def lstm_main():
    """
    Train and test the LSTM model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    test_size = config["test_size"]
    # ngram_range = tuple(config['ngram_range'])
    data_type = config["data_type"]
    max_len = config["max_len"]
    weighted_loss = config["weighted_loss"]
    n_epochs = config["n_epochs"]
    to_save = PACKAGE_ROOT_PATH.parent / "results" / "lstm" / data_type
    data_path = PACKAGE_ROOT_PATH.parent / "data" / "clean_data.csv"
    X_train, X_test, y_train, y_test, labels = sequence_data(data_path, test_size=test_size, data_type=data_type, max_len=max_len)  # type: ignore

    # Build the model
    vocab_size = 10000
    embedding_dim = 16
    lstm_out = 32
    batch_size = 64

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len)
    )
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_out)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(7, activation="softmax"))

    logger.info(model.summary())

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    if weighted_loss:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(zip(np.unique(y_train), class_weights))
        logger.info(f"Class weights: {class_weights}")
        to_save = to_save / "weighted_loss"
        history = model.fit(
            X_train,
            y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            class_weight=class_weights,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
        )

    # Plot the history
    plot_history(history, "LSTM", to_save=to_save)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    logger.info(f"Training Accuracy: {accuracy:.4f}")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    logger.info(f"Testing Accuracy:  {accuracy:.4f}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    y_pred = pd.Series(y_pred)
    y_pred = pd.DataFrame(
        y_pred.map(
            {
                0: "automobile",
                1: "entertainment",
                2: "politics",
                3: "science",
                4: "sports",
                5: "technology",
                6: "world",
            }
        )
    )
    y_test = pd.DataFrame(y_test.map({0: "automobile", 1: "entertainment", 2: "politics", 3: "science", 4: "sports", 5: "technology", 6: "world"}))  # type: ignore
    labels = np.array(
        [
            "automobile",
            "entertainment",
            "politics",
            "science",
            "sports",
            "technology",
            "world",
        ]
    )
    # Confusion matrix
    confusion_matrix(y_test, y_pred, "LSTM", labels=labels, to_save=to_save)  # type: ignore

    # Classification report
    get_results(y_test, y_pred, "LSTM", labels=labels, to_save=to_save)  # type: ignore

    return


if __name__ == "__main__":
    lstm_main()
