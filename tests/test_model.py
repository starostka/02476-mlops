import torch

# from src.models.dataset import MyDataset
from src.models.model import GCN
from tests import _PATH_DATA


def test_model_output_shape():
    dataset = torch.load(_PATH_DATA)
    data = dataset[0]
    model = GCN(hidden_channels=16)
    model.eval()

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    assert pred.shape == torch.Size([2708])  # predictions for all 2708 nodes
