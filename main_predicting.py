"""Map the six lunar oxides from an eight-band raster in geographic degrees."""

import argparse
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from ruamel.yaml import YAML
import torch
from tqdm import tqdm

from models import OneDCNN
from utils.spectra import ALL_OXIDES, compute_features, load_dataset_from_csv


NODATA_IN = -32768
NODATA_OUT = -9999.0
SCALE = 2e-5


def parse_args():
    parser = argparse.ArgumentParser(description='Predict six oxides by longitude block and write one GeoTIFF per oxide')
    parser.add_argument('--vrt', required=True, type=Path, help='Eight-band VRT or GeoTIFF')
    parser.add_argument('--datapath', type=Path, default=Path('datasets'), help='Directory containing the training data.csv')
    parser.add_argument('--modeldir', type=Path, default=Path('outputs/training'),
                        help='Parent directory of six training runs or six selected checkpoints')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/predictions'))
    parser.add_argument('--gpu', default='0', help='CUDA device index, or cpu')
    parser.add_argument('--blocks', default='all', help='Longitude block indices 1-9, e.g. 1,3,5 or all')
    parser.add_argument('--block-size', '--block_size', type=int, default=512, help='Read/write window side length in pixels')
    args = parser.parse_args()
    try:
        args.blocks = list(range(1, 10)) if args.blocks == 'all' else list(dict.fromkeys(
            int(value) for value in args.blocks.split(',')))
    except ValueError:
        parser.error('--blocks must be all or comma-separated integers')
    if args.block_size < 1 or any(b < 1 or b > 9 for b in args.blocks):
        parser.error('Window size must be positive and block indices must be between 1 and 9')
    return args


def select_checkpoint(root, oxide):
    runs = list(root.glob(f'log_OneDCNN_{oxide}_*'))
    if runs:
        # Run names end in the creation timestamp, regardless of epoch count or augmentation.
        def timestamp(path):
            match = re.search(r'(\d{8}_\d{6}(?:_\d{6})?)$', path.name)
            return match.group() if match else ''

        run = max(runs, key=timestamp)
        weights = sorted(run.glob('*best*IdxCV=*.pth'),
                         key=lambda p: int(p.stem.rsplit('=', 1)[1]))
        if not weights:
            raise FileNotFoundError(f'{run} is missing best cross-validation checkpoints')
        record = run / 'Records_GT-Test_Pred_CV.csv'
        if record.exists():
            result = pd.read_csv(record)
            fold = (result['GT'] - result['Pred']).abs().idxmin() + 1
            best = run / f'model_{oxide}_weights_best_IdxCV={fold}.pth'
            if best.is_file():
                return best
        return weights[0]
    # Historical checkpoint collections sometimes spell the calcium folder log_Cao.
    for name in ([oxide, 'Cao'] if oxide == 'CaO' else [oxide]):
        path = root / f'log_{name}' / 'model_best.pth'
        if path.is_file():
            return path
    raise FileNotFoundError(f'{root} is missing {oxide} training directory or log_{oxide}/model_best.pth')


def load_model_and_scaler(oxide, root, datapath, device):
    path = select_checkpoint(root, oxide)
    with path.with_name('config.yaml').open(encoding='utf-8') as src:
        config = YAML(typ='safe').load(src)
    if config['data']['dataname'] != oxide:
        raise ValueError(f'{path} configuration does not match {oxide} mismatch')
    model = OneDCNN().to(device)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['state_dict'])
    model.eval()
    scaler_x = scaler_y = None
    if config['data']['do_scale']:
        _, _, scaler_x, scaler_y = load_dataset_from_csv(oxide, datapath)
    print(f'{oxide}  {path}')
    return model, scaler_x, scaler_y


def predict_tile(chunk, models, device):
    valid = np.all(chunk != NODATA_IN, axis=0) & np.any(chunk != 0, axis=0)
    result = np.full((len(ALL_OXIDES), *valid.shape), NODATA_OUT, dtype=np.float32)
    if not valid.any():
        return result
    spectra = chunk[:, valid].T.astype(np.float32) * SCALE
    predictions = []
    with torch.inference_mode():
        for oxide in ALL_OXIDES:
            model, scaler_x, scaler_y = models[oxide]
            features = compute_features(spectra, oxide)
            if scaler_x is not None:
                features = scaler_x.transform(features)
            inputs = torch.from_numpy(features).unsqueeze(1)
            pred = np.concatenate([model(batch.to(device)).cpu().numpy()
                                   for batch in inputs.split(2048)])
            values = scaler_y.inverse_transform(pred) if scaler_y is not None else pred * 100
            predictions.append(np.clip(values[:, 0], 0, None))
    values = np.stack(predictions)
    total = values.sum(axis=0)
    finite = np.isfinite(total)
    rows, cols = np.where(valid)
    result[:, rows[finite], cols[finite]] = values[:, finite] / np.where(
        total[finite] > 0, total[finite], 1.0) * 100
    return result


def process_block(number, src, models, out_dir, size, device):
    west = -180 + (number - 1) * 40
    window = from_bounds(west, -65, west + 40, 65, src.transform)
    left = max(0, round(window.col_off))
    top = max(0, round(window.row_off))
    width = min(src.width, round(window.col_off + window.width)) - left
    height = min(src.height, round(window.row_off + window.height)) - top
    if width <= 0 or height <= 0:
        return 0
    extent = Window(left, top, width, height)
    profile = dict(driver='GTiff', dtype='float32', width=width, height=height, count=1,
                   crs=src.crs, transform=src.window_transform(extent), nodata=NODATA_OUT,
                   compress='deflate', tiled=True, blockxsize=256, blockysize=256,
                   BIGTIFF='IF_SAFER')
    with ExitStack() as stack:
        outputs = []
        for oxide in ALL_OXIDES:
            path = out_dir / f'{oxide}_block{number:02d}_{west:+d}to{west + 40:+d}.tif'
            outputs.append(stack.enter_context(rasterio.open(path, 'w', **profile)))
        for y in tqdm(range(0, height, size), desc=f'Block {number}'):
            for x in range(0, width, size):
                w, h = min(size, width - x), min(size, height - y)
                tile = src.read(indexes=range(1, 9), window=Window(left + x, top + y, w, h))
                result = predict_tile(tile, models, device)
                for band, dst in zip(result, outputs):
                    dst.write(band, 1, window=Window(x, y, w, h))
    return len(outputs)


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != 'cpu' and torch.cuda.is_available() else 'cpu')
    with rasterio.open(args.vrt) as src:
        if src.count != 8 or not src.crs or not src.crs.is_geographic:
            raise ValueError('Input must be an eight-band raster in geographic coordinates')
        if src.transform.a <= 0 or src.transform.e >= 0 or src.transform.b or src.transform.d:
            raise ValueError('Input raster must be north-up without rotation')
        models = {ox: load_model_and_scaler(ox, args.modeldir, args.datapath, device)
                  for ox in ALL_OXIDES}
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out_dir = args.output_dir / f'pred_blocks_{stamp}'
        out_dir.mkdir(parents=True)
        files = sum(process_block(n, src, models, out_dir, args.block_size, device)
                    for n in args.blocks)
        if files == 0:
            raise ValueError('Selected longitude blocks do not intersect the input image')
        print(f'{files} GeoTIFF -> {out_dir}')


if __name__ == '__main__':
    main()
