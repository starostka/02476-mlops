import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(1433, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 7)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x