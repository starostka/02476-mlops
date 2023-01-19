import os.path

import pytest
from tests import _PATH_DATA
from src.utilities.helpers import load_last_predictions, load_train_data
import pandas as pd


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_monitoring():

    last_predictions = load_last_predictions(samples=10)
    train_dataset = load_train_data(_PATH_DATA)

    assert isinstance(last_predictions, pd.DataFrame)
    assert isinstance(train_dataset, pd.DataFrame)
