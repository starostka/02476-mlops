import os
import warnings

import torch
import omegaconf
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import pytest
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src.models.model import GCN
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_training():
    # ignoring annoying and irrelevant warnings from lightning
    warnings.filterwarnings("ignore", category=PossibleUserWarning)

    cfg = omegaconf.OmegaConf.load('conf/config.yaml')

    model = GCN(
        hidden_channels=cfg.hyperparameters.hidden_channels,
        learning_rate=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay,
    )

    data = torch.load(_PATH_DATA)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, DataLoader(data), DataLoader(data))
