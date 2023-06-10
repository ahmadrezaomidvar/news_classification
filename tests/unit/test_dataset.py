"""Test the dataset."""

from news.dataset import (
    load_data,
    undersample_data,
    split_data,
    tfidf_vectorizer,
    smote_data,
)
from news.constant import PACKAGE_ROOT_PATH


class TestDataset:
    """Test the dataset."""

    def setup_method(self):
        """Setup the config for the test."""

        self.data_path = str(
            PACKAGE_ROOT_PATH.parent / "tests" / "fixture" / "data" / "clean_data.csv"
        )
        self.config = {
            "test_size": 0.2,
            "ngram_range": (1, 1),
            "data_type": "benchmark",
            "max_len": 70,
            "weighted_loss": False,
            "n_epochs": 10,
        }

    def test_load_data(self):
        """Test the load_data function."""
        df = load_data(self.data_path)
        assert df.shape == (1969, 3)

    def test_undersample_data(self):
        """Test the undersample_data function."""
        df = load_data(self.data_path)
        df = undersample_data(df)
        assert df.shape == (546, 3)

    def test_split_data(self):
        """Test the split_data function."""
        df = load_data(self.data_path)
        X_train, X_test, y_train, y_test = split_data(
            df, test_size=self.config["test_size"]
        )
        assert X_train.shape == (1575,)
        assert X_test.shape == (394,)
        assert y_train.shape == (1575,)
        assert y_test.shape == (394,)

    def test_tfidf_vectorizer(self):
        """Test the tfidf_vectorizer function."""
        df = load_data(self.data_path)
        X_train, X_test, y_train, y_test = split_data(
            df, test_size=self.config["test_size"]
        )
        X_train, X_test = tfidf_vectorizer(
            X_train, X_test, ngram_range=self.config["ngram_range"]
        )
        assert X_train.shape == (1575, 9526)
        assert X_test.shape == (394, 9526)

    def test_smote_data(self):
        """Test the smote_data function."""
        df = load_data(self.data_path)
        X_train, X_test, y_train, y_test = split_data(
            df, test_size=self.config["test_size"]
        )
        X_train, X_test = tfidf_vectorizer(
            X_train, X_test, ngram_range=self.config["ngram_range"]
        )
        X_train, y_train = smote_data(X_train, y_train)
        assert X_train.shape == (2464, 9526)
        assert y_train.shape == (2464,)
