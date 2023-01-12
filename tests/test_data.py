import os.path

import numpy as np
import pytest
import torch
from tests import _PATH_DATA


@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA),
    reason="Data files not found",
)
def test_dataset():
    dataset = torch.load(_PATH_DATA)
    assert len(dataset) == 1  # no of graphs
    assert dataset.num_features == 1433  # no of features
    assert dataset.num_classes == 7  # no of classes/labels


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_datapoint_shape():
    dataset = torch.load(_PATH_DATA)
    data = dataset[0]  # selct first and only graph

    assert data.x.shape == torch.Size([2708, 1433])
    assert data.edge_index.shape == torch.Size([2, 10556])
    assert data.y.shape == torch.Size([2708])
    assert data.train_mask.shape == torch.Size([2708])
    assert data.val_mask.shape == torch.Size([2708])
    assert data.test_mask.shape == torch.Size([2708])

    assert data.has_isolated_nodes() is False
    assert data.has_self_loops() is False
    assert data.is_undirected() is True


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
@pytest.mark.parametrize("split", ["train_mask", "val_mask", "test_mask"])
def test_all_labels_are_represented(split):
    dataset = torch.load(_PATH_DATA)
    data = dataset[0]

    labels = [0, 1, 2, 3, 4, 5, 6]
    for label in labels:
        assert np.isin(label, data.y[data[split]])
