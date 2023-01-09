import torch

from src.models.dataset import MyDataset
from src.models.model import MyAwesomeModel
from tests import _PATH_DATA


def test_model_output_shape():
    dataset = MyDataset(_PATH_DATA + "/test.pt")
    model = MyAwesomeModel()
    model.eval()

    input_, target = dataset[0]
    input_ = input_.unsqueeze(0)
    output = model(input_).argmax(dim=-1)
    assert output.shape == torch.Size([1])
