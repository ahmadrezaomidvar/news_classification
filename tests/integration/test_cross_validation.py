"""Test the integration of the package."""

from news.cross_validation import cv_main
from news.constant import PACKAGE_ROOT_PATH


class TestIntegrationCrossValidation:
    """
    Test the integration of the cross_validation.py module
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

    def test_cross_validation_main(self):
        """
        Test the cross_validation_main function
        """
        cv_main()
        assert True
