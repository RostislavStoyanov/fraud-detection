import numpy as np
import torch.utils.data as td


class OversamplingDataset(td.Dataset):
    def __init__(self, dataset_df, oversampler):

        y_orig = dataset_df['Class'].values.astype(np.float32)
        x_orig = dataset_df.drop('Class', axis=1).values.astype(np.float32)

        self.X, self.y = oversampler.fit_resample(x_orig, y_orig)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        return x, y
