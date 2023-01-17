import logging
import os

import matplotlib.pyplot as plt
import torch
import wandb
import hydra

from src.models.model import GCN

from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../conf", config_name="model")
def main(cfg: DictConfig) -> None:
    lr = cfg.hyperparameters.learning_rate
    wd = cfg.hyperparameters.weight_decay
    epochs = cfg.hyperparameters.epochs
    wandb_log = cfg.wandb
    model_checkpoint = cfg.checkpoint

    logger = logging.getLogger(__name__)

    if wandb_log:
        wandb.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("initializing model")
    model = GCN(hidden_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info("loading data")
    data = torch.load("data/processed/data.pt")[0]  # access first and only graph

    if wandb_log:
        wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info("starting training loop")
    loss_curve = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(
            data.x.to(device), data.edge_index.to(device)
        )  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask], data.y[data.train_mask].to(device)
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        loss_curve.append(loss.item())
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}; loss: {loss:.4f}")
            if wandb_log:
                wandb.log({'loss': loss})

    outdir = os.path.dirname(model_checkpoint)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    torch.save(model.state_dict(), model_checkpoint)

    plt.plot(loss_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plotfile = "training_curve.png"
    plt.savefig(os.path.join(outdir, plotfile))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
