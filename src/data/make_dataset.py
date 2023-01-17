# -*- coding: utf-8 -*-
import os
import logging
import torch
import shutil
import hydra

from pathlib import Path
from omegaconf import DictConfig
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


@hydra.main(version_base=None, config_path="../../conf", config_name="default")
def main(cfg: DictConfig) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting download")
    dataset = Planetoid(root="data", name="Cora", transform=NormalizeFeatures())
    logger.info("Download completed")

    outpath = os.path.join(cfg.dataset.output_filepath, "data.pt")
    torch.save(dataset, outpath)

    logger.info("Delete data folder created by Planetoid function")
    shutil.rmtree("data/Cora")
    logger.info("Raw data deleted")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
