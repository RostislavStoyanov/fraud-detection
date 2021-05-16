import numpy as np
import pandas as pd
import torch.utils.data as td


class OversamplingDataset(td.Dataset):
    def __init__(self, dataset_path, oversampler):
        df = pd.read_parquet(dataset_path)

        y_orig = df['Class'].astype(np.float32)
        x_orig = df.drop('Class', axis=1).astype(np.float32)
        print(x_orig.shape)

        self.X, self.y = oversampler.fit_resample(x_orig, y_orig)
        print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X.iloc[idx].values
        y = self.y.iloc[idx]

        return x, y
