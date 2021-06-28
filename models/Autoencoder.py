from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=30, out_features=50),
            nn.Sigmoid(),
            nn.Linear(in_features=50, out_features=25),
            nn.Sigmoid(),
            nn.Linear(in_features=25, out_features=15)
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=15, out_features=25),
            nn.Sigmoid(),
            nn.Linear(in_features=25, out_features=50),
            nn.Sigmoid(),
            nn.Linear(in_features=50, out_features=30)
        )

    def get_enc(self, x):
        return self.enc(x)

    def forward(self, x):
        encoded = self.enc(x)
        decoded = self.dec(encoded)

        return decoded
