from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=29, out_features=22),
            nn.Linear(in_features=22, out_features=15),
            nn.Linear(in_features=15, out_features=10)
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=10, out_features=15),
            nn.Linear(in_features=15, out_features=22),
            nn.Linear(in_features=22, out_features=29)
        )

    def get_enc(self, x):
        return self.enc(x)

    def forward(self, x):
        encoded = self.enc(x)
        decoded = self.dec(encoded)

        return decoded
