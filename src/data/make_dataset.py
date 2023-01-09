# -*- coding: utf-8 -*-
import logging
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
import wget
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    download_data(input_filepath)
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    for which in ["train", "test"]:
        files = build_filenames(which)
        content = []
        for filename in files:
            content.append(np.load(os.path.join(input_filepath, filename)))
        data = torch.tensor(np.concatenate([c["images"] for c in content])).reshape(
            -1, 1, 28, 28
        )
        target = torch.tensor(np.concatenate([c["labels"] for c in content]))
        # normalize using training set statistics
        if which == "train":
            mean = data.mean((0, 2, 3), keepdim=True)
            std = data.std((0, 2, 3), keepdim=True)
        data = (data - mean) / std
        # cast to single precision float
        data = data.float()
        outpath = os.path.join(output_filepath, f"{which}.pt")
        torch.save(
            {
                "data": data,
                "target": target,
            },
            outpath,
        )


def download_data(input_filepath):
    logger = logging.getLogger(__name__)
    filenames = build_filenames()
    for filename in filenames:
        filepath = os.path.join(input_filepath, filename)
        if not os.path.exists(filepath):
            dirpath = os.path.dirname(filepath)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            url = f"https://github.com/SkafteNicki/dtu_mlops/raw/main/data/{filename}"
            logger.info(f"downloading {url}")
            wget.download(url, out=filepath)
            sys.stdout.write("\n")
        else:
            logger.info(f"{filename} already downloaded")


def build_filenames(which="all"):
    train_filenames = [f"corruptmnist/train_{i}.npz" for i in range(5)]
    train_filenames += [f"corruptmnist_v2/train_{i+5}.npz" for i in range(3)]
    test_filenames = ["corruptmnist/test.npz"]
    if which == "all":
        output = train_filenames + test_filenames
    elif which == "train":
        output = train_filenames
    elif which == "test":
        output = test_filenames
    else:
        return ValueError(f"which must be train, test or all, got {which}")
    return output


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
