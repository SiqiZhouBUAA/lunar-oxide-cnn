# Model checkpoints

Mapping requires one checkpoint for each of SiO2, TiO2, Al2O3, FeO, MgO and CaO, using matching band definitions, angular features and units.

- Training output: pass the parent of the six `log_OneDCNN_<oxide>_...` directories to `--modeldir`. The latest run for each oxide is selected. The fold with the lowest absolute error in the per-sample CSV supplies its best checkpoint. If records or that checkpoint are unavailable, the lowest available fold index is used.
- Selected checkpoints: place `model_best.pth` and its training `config.yaml` in `model_ckpt/log_<oxide>/` and pass `--modeldir model_ckpt`. The legacy calcium directory `log_Cao` is also recognized.

The default scaling mode requires `--datapath` to point to the original training table. Changing the table changes input and output transformations. No pretrained CNN checkpoints are included.
