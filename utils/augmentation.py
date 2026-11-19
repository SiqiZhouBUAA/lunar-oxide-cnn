import numpy as np


def augment_data(x_train, y_train, n_bins=5, augment_factor=3, noise_level=0.02):
    """
    Oversample training data according to the target distribution.
    x_train: (N, D) float32
    y_train: (N, 1) float32
    Return augmented x_aug and y_aug.
    """
    y_flat = y_train.flatten()
    y_min, y_max = y_flat.min(), y_flat.max()

    # Divide targets into n_bins equal-width bins and count samples.
    bin_edges  = np.linspace(y_min, y_max, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=np.float32)
    sample_bin = np.zeros(len(y_flat), dtype=int)

    for i in range(len(y_flat)):
        b = np.searchsorted(bin_edges[1:], y_flat[i])
        b = min(b, n_bins - 1)
        sample_bin[i] = b
        bin_counts[b] += 1

    # Bin weight = maximum count / bin count (larger weights for sparse bins).
    max_count  = bin_counts.max()
    bin_weight = np.divide(max_count, bin_counts, out=np.ones_like(bin_counts), where=bin_counts > 0)

    # Rounded replication count per sample, bounded by 1 and augment_factor.
    repeat_times = np.clip(
        np.round(bin_weight[sample_bin]).astype(int), 1, augment_factor
    )

    x_list = [x_train]
    y_list = [y_train]
    feat_std = x_train.std(axis=0) + 1e-8  # Per-feature standard deviation controls noise amplitude.

    for i, n_repeat in enumerate(repeat_times):
        for _ in range(n_repeat - 1):  # Original sample is retained; add n_repeat - 1 copies.
            noise = np.random.randn(1, x_train.shape[1]).astype(np.float32)
            noise *= feat_std * noise_level
            x_list.append(x_train[i:i+1] + noise)
            y_list.append(y_train[i:i+1])

    x_aug = np.concatenate(x_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)

    # Shuffle samples.
    idx = np.random.permutation(len(x_aug))
    return x_aug[idx], y_aug[idx]
