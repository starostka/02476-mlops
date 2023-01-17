import torch
import joblib
from src.models.model import GCN


class Pipeline():
    """
    Orchestrates the model pipeline.
    """
    def __init__(self) -> None:
        # FIXME setting debug constants
        self.config = {'ENV': None, 'DEVICE': 'cpu', 'MODEL_PATH': "models/trained_model.pt"}

    def initialize(self):
        # Initialize model with latest checkpoint
        self.model = GCN()
        self.model.load_state_dict(torch.load(self.config['MODEL_PATH'], map_location=torch.device(self.config['DEVICE'])))
        self.model.eval()
