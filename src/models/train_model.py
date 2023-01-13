import logging
import os

import click
import matplotlib.pyplot as plt
import torch
import wandb

from src.models.model import GCN

@click.command(context_settings={"show_default": True})
@click.option("--lr", default=0.01)
@click.option("--wd", default=5e4)
@click.option("--epochs", default=100)
@click.option("--wandb", "wandb_log", is_flag=True)
def main(lr, wd, epochs, wandb_log):
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
                wandb_log({"loss": loss})

    outdir = "models"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = "trained_model.pt"
    torch.save(model.state_dict(), os.path.join(outdir, outfile))

    plt.plot(loss_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plotfile = "training_curve.png"
    plt.savefig(os.path.join(outdir, plotfile))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
