import logging

import click
import torch
import wandb

from src.models.dataset import MyDataset
from src.models.model import MyAwesomeModel


@click.command()
@click.argument('model_checkpoint', type=click.Path(exists=True))
@click.argument('test_set', type=click.Path(exists=True))
@click.option("--wandb", "wandb_log", is_flag=True)
def main(model_checkpoint, test_set, wandb_log):
    logger = logging.getLogger(__name__)

    if wandb_log:
        wandb.init()

    logger.info('loading model')
    model = MyAwesomeModel()
    state = torch.load(model_checkpoint)
    model.load_state_dict(state)
    model.eval()

    logger.info('loading data')
    dataset = MyDataset(test_set)
    input_, target = dataset[:]

    logger.info('predicting')
    output = model(input_).argmax(dim=-1)
    acc = torch.sum(output == target)/len(output)

    logger.info(f'accuracy: {acc:.2f}')

    if wandb_log:
        columns = ["id", "image", "guess", "truth"]
        data = [
            [i, wandb.Image(input_[i]), output[i], target[i]]
            for i in range(len(dataset))
        ]
        table = wandb.Table(data=data, columns=columns)
        wandb.log({'prediction_table': table})


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
