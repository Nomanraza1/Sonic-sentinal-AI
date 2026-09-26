import numpy as np


def augment(y, sr):
    yield y
    yield np.roll(y, int(sr * 0.08))
    yield y * 0.8
    yield y + np.random.default_rng(7).normal(0, 0.003, len(y))
