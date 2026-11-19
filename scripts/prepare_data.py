"""Prepare a training table from eight reflectance bands and oxide abundances."""

import argparse
from pathlib import Path

import pandas as pd

from utils.spectra import ALL_OXIDES, ANGLE_COLS, BAND_COLS, compute_features


def main():
    parser = argparse.ArgumentParser(description='Compute angular features from reflectance and prepare a training table')
    parser.add_argument('input', type=Path, help='Input CSV with bands expressed as reflectance')
    parser.add_argument('--output', type=Path, default=Path('datasets/data.csv'))
    parser.add_argument('--split', action='store_true', help='Also save six legacy single-oxide CSV files')
    args = parser.parse_args()
    outputs = [args.output.resolve()]
    if args.split:
        outputs.extend(args.output.with_name(f'{oxide}.csv').resolve() for oxide in ALL_OXIDES)
    if args.input.resolve() in outputs or len(set(outputs)) != len(outputs):
        parser.error('Input, output training table and split files must have distinct paths')
    table = pd.read_csv(args.input)
    spectra = table[BAND_COLS].to_numpy(dtype=float)
    for oxide, name in ANGLE_COLS.items():
        table[name] = compute_features(spectra, oxide)[:, -1]
    targets = [ox for ox in ALL_OXIDES if ox in table]
    if not targets or (args.split and len(targets) != len(ALL_OXIDES)):
        parser.error('Training requires at least one oxide target; --split requires all six')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    if args.split:
        names = ['band415', 'band750', 'band900', 'band950',
                 'band1001', 'band1050', 'band1250', 'band1550']
        for oxide in targets:
            subset = table[[*BAND_COLS, *ANGLE_COLS.values(), oxide]]
            subset.rename(columns=dict(zip(BAND_COLS, names))).to_csv(
                args.output.with_name(f'{oxide}.csv'), index=False)
    print(f'{len(table)} samples -> {args.output}')


if __name__ == '__main__':
    main()
