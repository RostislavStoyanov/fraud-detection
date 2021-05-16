from torch import nn


class HiddenReprClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        return self.classifier(x)
