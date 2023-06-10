from pathlib import Path
import logging
from typing import Tuple, Dict
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dataframe_image as dfi
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Logging and seed
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
seed = 1221
np.random.seed(seed)


def confusion_matrix(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    model_name: str,
    labels: NDArray = np.array(None),
    to_save: str = str(None),
) -> None:
    """
    Plot the confusion matrix.

    Parameters
    ----------
    y_true : pd.DataFrame
        True labels.
    y_pred : pd.DataFrame
        Predicted labels.
    model_name : str
        Name of the model.
    labels : NDArray, optional
        List of labels, by default None
    to_save : str, optional
        Path to save the confusion matrix, by default None

    Returns
    -------
    None
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, cmap=plt.cm.Blues, xticks_rotation=45.0
    )
    fig = disp.ax_.get_figure()
    fig.set_figwidth(11)
    fig.set_figheight(10)
    disp.ax_.set_title(f"{model_name}")
    disp.ax_.set_xlabel("Predicted label")
    disp.ax_.set_ylabel("True label")
    to_save = Path(to_save)  # type: ignore
    to_save.mkdir(exist_ok=True, parents=True)  # type: ignore
    plt.savefig(to_save / f"{model_name}_cm.png", dpi=300)  # type: ignore
    plt.close("all")
    logger.info(f"Confusion matrix for {model_name} saved.")
    return


def get_results(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    model_name: str,
    labels: NDArray = np.array(None),
    to_save: str = str(None),
) -> None:
    """
    Get the classification report and save it as a dataframe.

    Parameters
    ----------
    y_true : pd.DataFrame
        True labels.
    y_pred : pd.DataFrame
        Predicted labels.
    model_name : str
        Name of the model.
    labels : NDArray, optional
        List of labels, by default None
    to_save : str, optional
        Path to save the classification report, by default None

    Returns
    -------
    None
    """
    model_report_df = pd.DataFrame(
        classification_report(y_true, y_pred, labels=labels, output_dict=True)
    ).transpose()
    model_report_df.reset_index(inplace=True)
    model_report_df["Evaluated_model"] = model_name
    model_report_df = model_report_df.style.background_gradient(
        cmap="Blues", subset=["precision", "recall", "f1-score"]
    )
    to_save = Path(to_save)  # type: ignore
    to_save.mkdir(exist_ok=True, parents=True)  # type: ignore
    dfi.export(model_report_df, to_save / f"{model_name}_report.png", dpi=300)  # type: ignore
    logger.info(f"Results for {model_name} saved.")
    return


def plot_history(
    history: Dict[str, Dict], model_name: str, to_save: str = str(None)
) -> None:
    """
    Plot the training and validation accuracy and loss.

    Parameters
    ----------
    history : dict
        History of the model.
    model_name : str
        Name of the model.
    to_save : str, optional
        Path to save the plots, by default None

    Returns
    -------
    None
    """
    acc = history.history["accuracy"]  # type: ignore
    val_acc = history.history["val_accuracy"]  # type: ignore
    loss = history.history["loss"]  # type: ignore
    val_loss = history.history["val_loss"]  # type: ignore
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "b", label="Training accuracy")
    plt.plot(epochs, val_acc, "r", label="Validation accuracy")
    plt.title(f"Training and validation accuracy for {model_name}")
    plt.legend()
    to_save = Path(to_save)  # type: ignore
    to_save.mkdir(exist_ok=True, parents=True)  # type: ignore
    plt.savefig(to_save / f"{model_name}_acc.png", dpi=300)  # type: ignore
    plt.close("all")
    logger.info(f"Accuracy plot for {model_name} saved.")
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(f"Training and validation loss for {model_name}")
    plt.legend()
    plt.savefig(to_save / f"{model_name}_loss.png", dpi=300)  # type: ignore
    plt.close("all")
    logger.info(f"Loss plot for {model_name} saved.")
    return
