from src.models.model import MyAwesomeModel
from src.models.dataset import MyDataset
from tests import _PATH_DATA
import torch
import pytest
import os


@pytest.mark.skipif(
    (
        not os.path.exists(_PATH_DATA + "/train.pt")
        or not os.path.exists(_PATH_DATA + "/test.pt")
    ),
    reason="Data files not found",
)
def test_model_output_shape():
    dataset = MyDataset(_PATH_DATA + "/test.pt")
    model = MyAwesomeModel()
    model.eval()

    input_, target = dataset[0]
    input_ = input_.unsqueeze(0)
    output = model(input_).argmax(dim=-1)
    assert output.shape == torch.Size([1])
