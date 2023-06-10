"""Test the integration of the package."""

from news.lstm import lstm_main
from news.constant import PACKAGE_ROOT_PATH


class TestIntegrationLSTM:
    """
    Test the integration of the lstm.py module
    """

    def setup_method(self):
        """
        Setup the config for the test
        """
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

    def test_lstm_main(self):
        """
        Test the lstm_main function
        """
        lstm_main()
        assert True
