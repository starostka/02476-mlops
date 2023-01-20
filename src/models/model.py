import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(pl.LightningModule):
    def __init__(self, hidden_channels, learning_rate, weight_decay):
        super().__init__()
        self.conv1 = GCNConv(1433, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 7)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def train_or_val_step(self, batch, which):
        if which == "train":
            mask = batch.train_mask
        elif which == "val":
            mask = batch.val_mask
        else:
            raise ValueError(f"which must be train or val, got {which}")
        out = self(batch.x, batch.edge_index)
        loss = self.criterion(out[mask], batch.y[mask])
        self.log(f"{which}_loss", loss, batch_size=out.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self.train_or_val_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.train_or_val_step(batch, "val")

    def predict(self, data, index):
        # single prediction from index in dataset
        self.eval()
        out = self(data.x, data.edge_index)
        pred = out[index].argmax()  # class with highest prob

        # For some reason, i couldn't find ANY reliable reference for what the
        # correct label mapping is! I found the mapping below from a freaking
        # medium post...
        label_dict = {
            0: "Theory",
            1: "Reinforcement_Learning",
            2: "Genetic_Algorithms",
            3: "Neural_Networks",
            4: "Probabilistic_Methods",
            5: "Case_Based",
            6: "Rule_Learning",
        }

        return label_dict[pred.item()], pred

    def evaluate(self, data):
        # accuracy over test examples in dataset
        self.eval()
        out = self(data.x, data.edge_index)
        pred = out[data.test_mask].argmax(dim=1)  # class with highest prob
        acc = (pred == data.y[data.test_mask]).sum() / len(pred)
        return acc.item()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
