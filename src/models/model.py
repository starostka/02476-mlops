from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 20 * 20, 128),
            nn.Dropout(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
