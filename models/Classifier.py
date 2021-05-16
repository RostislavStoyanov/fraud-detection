from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=29, out_features=15),
            nn.ReLU(),
            nn.Linear(in_features=15, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=2)
        )

    def forward(self, x):
        return self.classifier(x)
