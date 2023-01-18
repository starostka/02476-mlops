import logging
import hydra
import torch

from omegaconf import DictConfig
from src.models.model import GCN
import time
from google.cloud import bigquery


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # wandb_log = cfg.wandb
    model_checkpoint = cfg.checkpoint
    logger = logging.getLogger(__name__)

    logger.info("loading model")
    model = GCN(hidden_channels=16)
    state = torch.load(model_checkpoint)
    model.load_state_dict(state)
    model.eval()

    logger.info("loading data")
    data = torch.load(cfg.dataset)[0]  # access first and only graph

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

    # for monitoring
    data_x = data.x[data.test_mask]
    data_y = data.y[data.test_mask]
    data_y_hat = pred[data.test_mask]

    client = bigquery.Client()
    table_id = "hybrid-essence-236114.model_prediction_log.model_prediction_log"
    print(client)
    import pandas as pd

    for x, y, y_hat in zip(data_x, data_y, data_y_hat):
        row_to_insert = [
            {
                "TIMESTAMP": time.time(),
                "INPUT": pd.DataFrame(x.numpy()).to_json(orient="values"),
                "OUTPUT": y.numpy().item(),
                "LABEL": y_hat.numpy().item(),
            }
        ]
        errors = client.insert_rows_json(table_id, row_to_insert)
        if errors:
            logger.info(f"Error: {str(errors)}")
        break


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
