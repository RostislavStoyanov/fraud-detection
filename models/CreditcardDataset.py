import torch.utils.data as td
import numpy as np


class CreditcardDataset(td.Dataset):
    def __init__(self, dataset_df):

        self.y = dataset_df['Class'].astype(np.float32)
        self.X = dataset_df.drop('Class', axis=1).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X.iloc[idx].values
        y = self.y.iloc[idx]

        return x, y
