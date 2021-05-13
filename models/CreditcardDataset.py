import torch.utils.data as td
import pandas as pd


class CreditcardDataset(td.Dataset):
    def __init__(self, df):
        self.y = df['Class']
        self.X = df.drop('Class')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X.iloc[idx].values()
        y = self.y.iloc[idx]

        return x, y
