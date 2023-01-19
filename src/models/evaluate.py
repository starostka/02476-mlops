import logging
import hydra
import torch

from omegaconf import DictConfig
from src.models.model import GCN

# import time
# from google.cloud import bigquery


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # wandb_log = cfg.wandb
    logger = logging.getLogger(__name__)

    logger.info("loading model")
    model = GCN(
        hidden_channels=cfg.hyperparameters.hidden_channels,
        learning_rate=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay,
    )
    state = torch.load(cfg.checkpoint)
    model.load_state_dict(state["state_dict"])

    logger.info("loading data")
    data = torch.load(cfg.dataset)[0]  # access first and only graph

    logger.info("evaluating")
    acc = model.evaluate(data)
    logger.info(f"accuracy: {acc:.2f}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
