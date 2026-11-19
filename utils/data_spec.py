import os
import numpy as np
from torch.utils import data
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_dataset(dataname, datapath, do_scale=False):
    data_df = pd.read_csv(os.path.join(datapath, 'data_Kaguya_{}.csv'.format(dataname)))

    data_np = data_df.to_numpy()

    spec_data = data_np[:, :8].astype(np.float32)   # Select the eight spectral bands.
    gt_content = data_np[:, -1, np.newaxis].astype(np.float32)  # The final column contains the Al2O3 target.

    if do_scale:
        scaler_X = StandardScaler().fit(spec_data)
        scaler_Y = StandardScaler().fit(gt_content)
        spec_data = scaler_X.transform(spec_data)
        gt_content = scaler_Y.transform(gt_content)
        return spec_data, gt_content, scaler_X, scaler_Y
    else:
        gt_content *= 0.01
        return spec_data, gt_content


def generate_dataset(spec_data, gt, idx_sample, num_samples):
    idx_train = np.ones(num_samples).astype(bool)
    idx_train[idx_sample] = False
    x_test = spec_data[np.newaxis, idx_sample, :]
    y_test = gt[np.newaxis, idx_sample]
    x_train = spec_data[idx_train]
    y_train = gt[idx_train]

    return x_train, y_train, x_test, y_test


class DataSetGeneratorSpec(data.Dataset):
    """Generates data"""
    def __init__(self, data, targets=None, transforms=None):
        """Initialization"""
        self.data = data

        self.transforms = transforms
        if targets is None:
            self.has_target = False
        else:
            self.targets = targets
            self.has_target = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Generate one sample of data"""
        x_data = self.data[np.newaxis, index]
        if self.transforms is None:
            x_data = torch.from_numpy(x_data)
        else:
            x_data = self.transforms(x_data).squeeze(dim=0)

        if self.has_target:
            label = self.targets[index]
            return x_data, label
        else:
            return x_data
