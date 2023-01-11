import logging

import click
import torch
from src.models.model import GCN


@click.command()
@click.argument("model_checkpoint", type=click.Path(exists=True))
@click.option("--wandb", "wandb_log", is_flag=True)
def main(model_checkpoint, wandb_log):
    logger = logging.getLogger(__name__)

    logger.info("loading model")
    model = GCN(hidden_channels=16)
    state = torch.load(model_checkpoint)
    model.load_state_dict(state)
    model.eval()

    logger.info("loading data")
    data = torch.load("data/processed/data.pt")[0]  # access first and only graph

    logger.info("predicting")
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = (
        pred[data.test_mask] == data.y[data.test_mask]
    )  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(
        data.test_mask.sum()
    )  # Derive ratio of correct predictions.

    logger.info(f"accuracy: {test_acc:.2f}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
