from src.models.model import GCN

class Pipeline():
    """
    Orchestrates the model pipeline.
    """
    def __init__(self, model=GCN) -> None:
        self.model = model
    