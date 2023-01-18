import os
import warnings

import matplotlib.pyplot as plt
import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loggers import WandbLogger

from src.models.model import GCN
from src.models.callbacks import MetricsCallback


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ignoring annoying and irrelevant warnings from lightning
    warnings.filterwarnings("ignore", category=PossibleUserWarning)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GCN(
        hidden_channels=cfg.hyperparameters.hidden_channels,
        learning_rate=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay,
    ).to(device)

    data = torch.load(cfg.dataset)

    # setup trainer; thanks lightning for the "minimum" boilerplate!
    if os.path.exists(cfg.checkpoint):
        os.remove(cfg.checkpoint)
    ckpt_dir = os.path.dirname(cfg.checkpoint)
    ckpt_filename = os.path.basename(cfg.checkpoint)
    if ckpt_filename.endswith('.ckpt'):
        ckpt_filename = ckpt_filename[:-5]
    metrics_callback = MetricsCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_filename,
        monitor='val_loss',
    )
    if cfg.wandb:
        logger = WandbLogger(
            project="Pytorch Geometric Model",
            entity="02476-mlops-12",
        )
    else:
        logger = True
    trainer = pl.Trainer(
        max_epochs=cfg.hyperparameters.epochs,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[metrics_callback, checkpoint_callback]
    )
    trainer.fit(model, DataLoader(data), DataLoader(data))

    # plot training curve
    plt.plot(metrics_callback.train_loss)
    plt.plot(metrics_callback.val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(ckpt_dir, "training_curve.png"))
    # plt.show()


if __name__ == "__main__":
    main()
