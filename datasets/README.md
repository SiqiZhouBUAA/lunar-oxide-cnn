# Input data

Place a UTF-8 `data.csv` in the directory passed to `--datapath`. Rows correspond to sample indices in training outputs.

| Columns | Meaning |
| --- | --- |
| `414nm-1`, `749nm-2`, `901nm-3`, `950nm-4`, `1001nm-5`, `1049nm-7`, `1248nm-8`, `1548nm-9` | Dimensionless reflectance, in MI band order 1, 2, 3, 4, 5, 7, 8, 9 |
| `theta_Ti`, `theta_Fe` | Angular features in radians |
| `SiO2`, `TiO2`, `Al2O3`, `FeO`, `MgO`, `CaO` | Oxide concentrations in wt% |

Additional metadata columns may be retained. Required inputs and the selected target must be finite numeric values. Inference must use the original training table to reconstruct scaling.

```bash
python -m scripts.prepare_data raw_samples.csv --output datasets/data.csv
```

Mapping requires a north-up, unrotated eight-band VRT or GeoTIFF in lunar geographic coordinates, stacked in the order above. Raw raster values are multiplied by 0.00002 to obtain reflectance. Any band equal to -32768, or all bands equal to zero, marks an invalid pixel. VRT source files must remain accessible. Source imagery is not included. CSV and imagery patterns may be ignored by Git; explicitly include the intended release data when committing.
