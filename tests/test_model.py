import os

import omegaconf
import pytest
import torch

from src.models.model import GCN
from tests import _PATH_CONF, _PATH_DATA


def _init_dataset_and_model():
    data = torch.load(_PATH_DATA)[0]

    cfg = omegaconf.OmegaConf.load(_PATH_CONF)
    model = GCN(
        hidden_channels=cfg.hyperparameters.hidden_channels,
        learning_rate=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay,
    )
    model.eval()

    return data, model


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_model_output_shape():
    data, model = _init_dataset_and_model()
    out = model(data.x, data.edge_index)
    assert out.shape == torch.Size([2708, 7])  # predictions for all 2708 nodes


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_model_evaluate():
    data, model = _init_dataset_and_model()
    acc = model.evaluate(data)
    assert isinstance(acc, float)


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_model_predict():
    data, model = _init_dataset_and_model()
    pred, pred_int = model.predict(data, 0)
    assert isinstance(pred, str)
    assert isinstance(pred_int.item(), int)
