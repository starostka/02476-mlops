import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_path):
        content = torch.load(tensor_path)
        self._data = content['data']
        self._target = content['target']

    def __getitem__(self, idx):
        return self._data[idx], self._target[idx]

    def __len__(self):
        return len(self._data)
