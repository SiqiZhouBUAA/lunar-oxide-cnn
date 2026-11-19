"""Kaguya MI feature order and concentration units used by the OneDCNN runs."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BAND_COLS = ['414nm-1', '749nm-2', '901nm-3', '950nm-4',
             '1001nm-5', '1049nm-7', '1248nm-8', '1548nm-9']
ALL_OXIDES = ('SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO')
ANGLE_COLS = {'TiO2': 'theta_Ti', 'FeO': 'theta_Fe'}
FEATURE_COLS = {
    oxide: BAND_COLS + ([ANGLE_COLS[oxide]] if oxide in ANGLE_COLS else [])
    for oxide in ALL_OXIDES
}


def compute_features(bands, oxide):
    """Append the Ti or Fe angle to reflectance spectra in BAND_COLS order."""
    if oxide not in ANGLE_COLS:
        return bands
    r749 = bands[:, 1]
    safe_r749 = np.where(np.abs(r749) > 1e-8, r749, 1e-8)
    if oxide == 'TiO2':
        numerator = bands[:, 0] / safe_r749 - 0.208
        denominator = r749 + 0.108
    else:
        numerator = bands[:, 3] / safe_r749 - 1.250
        denominator = r749 - 0.037
    denominator = np.where(np.abs(denominator) > 1e-8, denominator, 1e-8)
    angle = np.arctan(numerator / denominator)
    return np.column_stack([bands, angle])


def load_dataset_from_csv(dataname, datapath, do_scale=True):
    frame = pd.read_csv(Path(datapath) / 'data.csv')
    columns = FEATURE_COLS[dataname]
    missing = set([*columns, dataname]).difference(frame.columns)
    if missing:
        raise ValueError(f"data.csv is missing columns: {', '.join(sorted(missing))}")
    features = frame[columns].to_numpy(dtype=np.float32)
    targets = frame[[dataname]].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f'{dataname} inputs or targets contain non-finite values')
    if not do_scale:
        return features, targets * 0.01
    scaler_x = StandardScaler().fit(features)
    scaler_y = StandardScaler().fit(targets)
    return scaler_x.transform(features), scaler_y.transform(targets), scaler_x, scaler_y
