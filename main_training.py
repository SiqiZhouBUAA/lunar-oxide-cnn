"""Train one oxide model per held-out sample using the original LOOCV protocol."""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import random

from easydict import EasyDict
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader

from models import OneDCNN
from trainer import test_model, train_model
from utils.augmentation import augment_data
from utils.data_spec import DataSetGeneratorSpec, generate_dataset
from utils.record import record_output
from utils.spectra import ALL_OXIDES, load_dataset_from_csv
from utils.utils import Logger


def boolean(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Use true or false')


def parse_args(augment_default=False):
    parser = argparse.ArgumentParser(description='Leave-one-out training for one lunar oxide')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('configs.yaml'))
    parser.add_argument('--gpu', help='CUDA device index, or cpu')
    parser.add_argument('--model-type', choices=['OneDCNN'])
    parser.add_argument('--dataname', choices=ALL_OXIDES)
    parser.add_argument('--datapath', help='Directory containing data.csv')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/training'))
    parser.add_argument('--do-scale', type=boolean)
    parser.add_argument('--save-ckpt', type=boolean)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--loss-type', choices=['MSE', 'L1'])
    parser.add_argument('--record-type', choices=['CV'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augment', type=boolean, nargs='?', const=True, default=augment_default)
    parser.add_argument('--augment-factor', type=int, default=3)
    parser.add_argument('--aug-bins', type=int, default=5)
    parser.add_argument('--noise-level', type=float, default=0.02)
    return parser.parse_args()


def load_config(args):
    with args.config.open(encoding='utf-8') as src:
        config = YAML(typ='safe').load(src)
    for section, options in (
        (config, {'epochs': args.epochs, 'batch_size': args.batch_size,
                  'record_type': args.record_type, 'save_ckpt': args.save_ckpt}),
        (config['model'], {'model_type': args.model_type}),
        (config['loss'], {'loss_type': args.loss_type}),
        (config['data'], {'dataname': args.dataname, 'datapath': args.datapath,
                          'do_scale': args.do_scale}),
        (config['optim'], {'base_lr': args.lr, 'weight_decay': args.wd,
                           'patience': args.patience}),
    ):
        section.update({key: value for key, value in options.items() if value is not None})
    if args.gpu is not None:
        config['use_gpu'] = args.gpu != 'cpu'
        if config['use_gpu']:
            config['gpu_id'] = int(args.gpu)
    if config['model']['model_type'] != 'OneDCNN':
        raise ValueError('The supported model implementation is OneDCNN')
    config.update(seed=args.seed, augment=args.augment, augment_factor=args.augment_factor,
                  aug_bins=args.aug_bins, noise_level=args.noise_level)
    return config


def train_fold(cfg, features, targets, index):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    oxide = cfg.data.dataname
    for kind in ('latest', 'best'):
        filename = f'model_{oxide}_weights_{kind}_IdxCV={index + 1}.pth'
        cfg[f'{kind}_ckpt'] = str(Path(cfg.work_dir) / filename)
    x_train, y_train, x_test, y_test = generate_dataset(features, targets, index, len(targets))
    if cfg.augment:
        x_train, y_train = augment_data(x_train, y_train, cfg.aug_bins,
                                        cfg.augment_factor, cfg.noise_level)
    loaders = {
        'train': DataLoader(DataSetGeneratorSpec(x_train, y_train), batch_size=cfg.batch_size,
                            shuffle=cfg.data.is_shuffle, num_workers=cfg.num_workers),
        'valid': DataLoader(DataSetGeneratorSpec(x_test, y_test), batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers),
    }
    model = OneDCNN().to(cfg.device)
    criterion = {'MSE': torch.nn.MSELoss, 'L1': torch.nn.L1Loss}[cfg.loss.loss_type]()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr,
                                 weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    best, _ = train_model(model, loaders, criterion, optimizer, scheduler, cfg)
    _, prediction = test_model(best, loaders['valid'], criterion, cfg)
    return prediction


def main(augment_default=False):
    args = parse_args(augment_default)
    config = load_config(args)
    cfg = EasyDict(config)
    if min(cfg.epochs, cfg.batch_size, cfg.aug_bins, cfg.augment_factor) < 1:
        raise ValueError('Epochs, batch size, bin count and augmentation factor must be positive')
    oxide = cfg.data.dataname
    features, targets, *scalers = load_dataset_from_csv(oxide, cfg.data.datapath, cfg.data.do_scale)
    if len(targets) < 2:
        raise ValueError('Leave-one-out cross-validation requires at least two samples')
    if scalers:
        cfg.scaler_X, cfg.scaler_Y = scalers
    cfg.device = torch.device(f'cuda:{cfg.gpu_id}' if cfg.use_gpu and torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    tag = f'_Aug{cfg.augment_factor}' if cfg.augment else ''
    work_dir = args.output_dir / f'log_OneDCNN_{oxide}_Epoch={cfg.epochs}{tag}_{stamp}'
    work_dir.mkdir(parents=True)
    cfg.work_dir = str(work_dir)
    config['in_channels'] = features.shape[1]
    with (work_dir / 'config.yaml').open('w', encoding='utf-8') as dst:
        YAML(typ='safe').dump(config, dst)
    cfg.logger = Logger(work_dir / f'{stamp}.log', logging.INFO, str(work_dir)).get_log()
    cfg.logger.info('%s | %s | %d samples | %d features', oxide, cfg.device, *features.shape)
    predictions = []
    for index in range(len(targets)):
        cfg.logger.info('Fold %d/%d', index + 1, len(targets))
        predictions.append(train_fold(cfg, features, targets, index))
    r2, rmse = record_output(targets, predictions, cfg)
    metrics = pd.DataFrame(np.vstack([rmse, r2]), index=['RMSE_CV', 'R2_CV'],
                           columns=[f'Result_{oxide}'])
    metrics.to_csv(work_dir / f'Records_RMSE-R2_{oxide}.csv')
    print(metrics)
    print(work_dir)


if __name__ == '__main__':
    main()
