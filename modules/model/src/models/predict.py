import argparse
import torch

import omegaconf
from src.models.model import GCN


def main():
    cfg = omegaconf.OmegaConf.load("conf/config.yaml")

    model = GCN(
        hidden_channels=cfg.hyperparameters.hidden_channels,
        learning_rate=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay,
    )
    state = torch.load(cfg.checkpoint)
    model.load_state_dict(state["state_dict"])

    data = torch.load(cfg.dataset)[0]  # access first and only graph
    pred = model.predict(data, args.index)

    print(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help="index of example to predict")
    args = parser.parse_args()

    main()
