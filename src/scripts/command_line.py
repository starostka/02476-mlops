import fire

from src.models.train_model import main as train_model


def welcome():
    print("MLOps Group 12: CLI Tool")


def main():
    fire.Fire({
        'welcome': welcome,
        'train': train_model
    })
