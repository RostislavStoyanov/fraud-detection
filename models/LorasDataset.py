from utils import loras_oversample_dataframe
import numpy as np
import torch.utils.data as td


class LorasDataset(td.Dataset):
    def __init__(self, dataset_df):
        dataset_df = dataset_df.astype(np.float32)

        self.X, self.y = loras_oversample_dataframe(dataset_df)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        return x, y
