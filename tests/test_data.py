from tests import _PATH_DATA
from src.models.dataset import MyDataset
import torch
import numpy as np
import os.path
import pytest


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_length():
    train_dataset = MyDataset(_PATH_DATA + "/train.pt")
    test_dataset = MyDataset(_PATH_DATA + "/test.pt")
    assert len(train_dataset) == 40000
    assert len(test_dataset) == 5000


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
@pytest.mark.parametrize("datapath", ["/train.pt", "/test.pt"])
def test_datapoint_shape(datapath):
    dataset = MyDataset(_PATH_DATA + datapath)
    X, y = dataset[0]
    assert X.shape == torch.Size([1, 28, 28])


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
@pytest.mark.parametrize("datapath", ["/train.pt", "/test.pt"])
def test_all_labels_are_represented(datapath):
    dataset = MyDataset(_PATH_DATA + datapath)
    y = dataset._target
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    for l in labels:
        assert np.isin(l, y)
