import numpy as np


def make_variants(y, sr, seed):
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(-int(sr * 0.18), int(sr * 0.18) + 1))
    noisy = y + rng.normal(0, 0.003, len(y))
    quieter = y * float(rng.uniform(0.72, 0.92))
    return {"shift": np.roll(y, shift), "noise": noisy, "volume": quieter}
