"""Plot measured and predicted oxide content for each LOOCV run."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser(description='Plot observed versus predicted values from leave-one-out results')
    parser.add_argument('--runs', type=Path, default=Path('outputs/training'))
    parser.add_argument('--output', type=Path, default=Path('outputs/regression.png'))
    args = parser.parse_args()
    records = sorted(args.runs.glob('log_OneDCNN_*/Records_GT-Test_Pred_CV.csv'))
    if not records:
        parser.error(f'{args.runs} contains no training records')
    cols = min(3, len(records))
    rows = (len(records) + cols - 1) // cols
    sns.set_style('ticks')
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.5 * rows),
                             constrained_layout=True, squeeze=False)
    palette = ['#4E79A7', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1']
    for i, (ax, path) in enumerate(zip(axes.flat, records)):
        oxide = path.parent.name.split('_')[2]
        frame = pd.read_csv(path)
        measured, predicted = frame['GT'], frame['Pred']
        metrics = path.with_name(f'Records_RMSE-R2_{oxide}.csv')
        if metrics.is_file():
            values = pd.read_csv(metrics, index_col=0)[f'Result_{oxide}']
            r2, rmse = values['R2_CV'], values['RMSE_CV']
        else:
            r2 = r2_score(measured, predicted)
            rmse = np.sqrt(mean_squared_error(measured, predicted))
        sns.regplot(x=measured, y=predicted, ax=ax, color=palette[i % len(palette)],
                    scatter_kws={'alpha': 0.5, 's': 25}, line_kws={'lw': 1.5}, seed=42)
        lo, hi = frame[['GT', 'Pred']].min().min(), frame[['GT', 'Pred']].max().max()
        margin = 0.05 * (hi - lo) or 0.05
        limits = (lo - margin, hi + margin)
        ax.plot(limits, limits, '--', color='gray', lw=1)
        ax.set(title=oxide, xlabel=f'Measured {oxide} (wt%)', ylabel=f'Predicted {oxide} (wt%)',
               xlim=limits, ylim=limits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.text(0.04, 0.96, f'$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$',
                transform=ax.transAxes, va='top')
    for ax in list(axes.flat)[len(records):]:
        ax.set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
