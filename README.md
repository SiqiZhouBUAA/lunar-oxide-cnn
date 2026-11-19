# Lunar Oxide CNN

A PyTorch 1D CNN maps eight Kaguya MI reflectance bands to lunar oxide concentrations (wt%). One model is trained per oxide. SiO2, Al2O3, MgO and CaO use eight bands; TiO2 and FeO additionally use theta_Ti and theta_Fe, respectively.

## Installation

Use Python 3 with the dependencies listed in `requirements.txt`. A compatible PyTorch installation is required; CUDA is optional.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

## Training

The supplied table is `datasets/data.csv`. See [data conventions](datasets/README.md). Defaults are in `configs.yaml`; command-line options override them.

```bash
python main_training.py --dataname Al2O3 --gpu cpu
python strengthen_training.py --dataname FeO --augment-factor 3 --gpu cpu
```

For a short execution check, add `--epochs 1`; this does not reproduce a converged model. Train all six targets before joint mapping. `strengthen_training.py` enables the same augmentation available through `main_training.py --augment`: target-bin oversampling and Gaussian noise are applied only to each training subset.

Training writes timestamped `log_OneDCNN_<oxide>_...` directories under `outputs/training`, including per-fold checkpoints, `Records_GT-Test_Pred_CV.csv` and `Records_RMSE-R2_<oxide>.csv`.

```bash
python -m scripts.plot_regression --runs outputs/training --output outputs/regression.png
```

## Mapping

Supply your own eight-band lunar raster and trained checkpoints. Neither pretrained CNN weights nor a demo raster is included.

```bash
python main_predicting.py --vrt datasets/kaguya_mi.vrt --blocks all --datapath datasets --modeldir outputs/training --gpu cpu
```

See [checkpoint selection](model_ckpt/README.md). Output comprises six GeoTIFFs per intersecting longitude block. Nine 40-degree blocks cover -180 to 180 degrees; latitude is restricted to -65 to 65 degrees. Windowed processing limits memory use. Negative predictions are clipped to zero; positive six-oxide totals are normalized to 100%; all-zero predictions remain zero. Output NoData is -9999.

## Evaluation interpretation

The original experimental procedure is preserved. By default, feature and target scalers are fitted on the full table before leave-one-out splitting, and the held-out sample selects the training epoch in each fold. Reported RMSE and R2 therefore describe this procedure; they are not an independent estimate of generalization. Mapping checkpoint selection also uses held-out errors. An independent test set is needed for independent validation.

## Files and attribution

`models.py` defines OneDCNN; `trainer.py` implements training and validation; `utils/` contains spectral features, augmentation and records; `scripts/` contains table preparation and plotting. The original DL-IM MIT notice is retained in [LICENSE](LICENSE). Repository organization follows [RLCM-DL](https://github.com/zyyan-cc/RLCM-DL).

## Manuscript and contact

**Regional variations in characteristic melting temperatures of lunar regolith revealed by Chang’e-6-constrained oxide mapping**

Publication status: Manuscript under submission.

Corresponding author: Feng Li, Beihang University.  
Email: [lifeng98@buaa.edu.cn](mailto:lifeng98@buaa.edu.cn).
