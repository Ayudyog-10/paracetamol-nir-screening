"""Spectral preprocessing used across the project (stateless, row-wise)."""
import numpy as np
from scipy.signal import savgol_filter


def snv(X):
    X = np.asarray(X, float)
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def sg(X, deriv=1, window=15, poly=2):
    return savgol_filter(np.asarray(X, float), window, poly, deriv=deriv, axis=1)


def msc(X, ref):
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for i, x in enumerate(X):
        b, a = np.polyfit(ref, x, 1)
        out[i] = (x - a) / b
    return out


PIPES = {
    "SNV": lambda X: snv(X),
    "SNV+SG1": lambda X: sg(snv(X), 1),
    "SG1+SNV": lambda X: snv(sg(X, 1)),
    "SNV+SG2": lambda X: sg(snv(X), 2, 21),
    "Raw": lambda X: np.asarray(X, float),
}


def apply(name, X, wl=None, lo=None, hi=None):
    Z = PIPES[name](X)
    if wl is not None and lo is not None:
        m = (wl >= lo) & (wl <= hi)
        Z = Z[:, m]
    return Z
