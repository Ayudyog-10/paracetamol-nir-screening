"""Readers for METASPEQ / Hamamatsu instrument exports."""
import io
import numpy as np
import pandas as pd


def _instrument_csv(text):
    lines = text.splitlines()
    hi = next(i for i, l in enumerate(lines) if l.startswith("Pixel,"))
    rows = [l.split(",") for l in lines[hi + 1:] if l[:1].isdigit()]
    a = np.array([[float(v) for v in r[:5]] for r in rows])
    wl, dark, ref, s = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    A = -np.log10(np.clip((s - dark) / (ref - dark), 1e-6, None))
    return A[None, :], wl


def _table(df):
    wcols = []
    for c in df.columns:
        try:
            v = float(str(c).replace("A_", ""))
            if 800 < v < 2600: wcols.append((c, v))
        except ValueError:
            pass
    if len(wcols) >= 50:
        return df[[c for c, _ in wcols]].astype(float).values, np.array([v for _, v in wcols])
    ab = [c for c in df.columns if str(c).startswith("Absorbance_")]
    if len(ab) == 256:
        return df[ab].astype(float).values, None
    raise ValueError("could not find wavelength columns (need headers like 892.43 or Absorbance_1..256)")


def read_any(src, name=None):
    """Return (X, wavelengths or None, kind). src = path or bytes."""
    name = name or (src if isinstance(src, str) else "upload")
    raw = open(src, "rb").read() if isinstance(src, str) else src
    if name.lower().endswith((".xlsx", ".xls")):
        X, wl = _table(pd.read_excel(io.BytesIO(raw)))
        return X, wl, "absorbance table"
    text = raw.decode("latin1")
    if "Pixel,WaveLength" in text:
        X, wl = _instrument_csv(text)
        return X, wl, "instrument raw CSV (dark/ref/sample)"
    X, wl = _table(pd.read_csv(io.StringIO(text)))
    return X, wl, "absorbance table"
