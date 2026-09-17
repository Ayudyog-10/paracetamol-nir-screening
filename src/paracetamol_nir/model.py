"""Screening model for paracetamol + talc blends.

Scan-level SVR + random-forest ensemble on SNV -> Savitzky-Golay 1st-derivative spectra,
with a PCA applicability-domain check (Hotelling T2 and Q residual) and a spectral
identity check against pure paracetamol.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .preprocess import snv, sg

LOW_CUT, HIGH_CUT, BORDER = 90.0, 110.0, 5.0   # % label claim screening bands


def prep(X):
    return sg(snv(X), 1)


def band(v):
    if v < LOW_CUT - BORDER:
        return "LOW"
    if v > HIGH_CUT + BORDER:
        return "HIGH"
    if LOW_CUT + BORDER <= v <= HIGH_CUT - BORDER:
        return "WITHIN RANGE"
    return "BORDERLINE - confirm by HPLC"


class BlendScreeningModel:
    def __init__(self, n_pc=10):
        self.n_pc = n_pc

    def fit(self, X, y, wavelengths, pure_api=None, pure_talc=None):
        self.wavelengths = np.asarray(wavelengths, float)
        Z = prep(X)
        self.svr = make_pipeline(StandardScaler(), SVR(C=100, gamma="scale", epsilon=0.5)).fit(Z, y)
        self.rf = RandomForestRegressor(n_estimators=150, max_features="sqrt", random_state=0, n_jobs=-1).fit(Z, y)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-12
        Zs = (Z - self.mu) / self.sd
        self.pca = PCA(n_components=self.n_pc).fit(Zs)
        t2, q = self._t2q(Z)
        self.t2_lim, self.q_lim = float(np.percentile(t2, 99)), float(np.percentile(q, 99))
        self.y_range = (float(np.min(y)), float(np.max(y)))
        self.ref_api = None if pure_api is None else snv(np.atleast_2d(pure_api)).mean(0)
        self.ref_talc = None if pure_talc is None else snv(np.atleast_2d(pure_talc)).mean(0)
        self.ref_blend = snv(X).mean(0)
        return self

    def _t2q(self, Z):
        Zs = (Z - self.mu) / self.sd
        T = self.pca.transform(Zs)
        t2 = np.sum(T ** 2 / self.pca.explained_variance_, 1)
        q = np.sum((Zs - self.pca.inverse_transform(T)) ** 2, 1)
        return t2, q

    def _check_axis(self, X, wavelengths):
        X = np.atleast_2d(np.asarray(X, float))
        if wavelengths is not None:
            w = np.asarray(wavelengths, float)
            if len(w) != len(self.wavelengths) or np.max(np.abs(w - self.wavelengths)) > 0.5:
                X = np.vstack([np.interp(self.wavelengths, w, x) for x in X])
        if X.shape[1] != len(self.wavelengths):
            raise ValueError(f"expected {len(self.wavelengths)} points, got {X.shape[1]}")
        return X

    def predict_scans(self, X, wavelengths=None):
        X = self._check_axis(X, wavelengths)
        Z = prep(X)
        p = 0.5 * (self.svr.predict(Z) + self.rf.predict(Z))
        t2, q = self._t2q(Z)
        S = snv(X)
        corr = lambda r: np.array([np.corrcoef(s, r)[0, 1] for s in S])
        return dict(pred=p, t2=t2, q=q, in_domain=(t2 <= self.t2_lim) & (q <= self.q_lim),
                    r_blend=corr(self.ref_blend),
                    r_api=corr(self.ref_api) if self.ref_api is not None else np.full(len(X), np.nan),
                    r_talc=corr(self.ref_talc) if self.ref_talc is not None else np.full(len(X), np.nan))

    def screen(self, X, wavelengths=None):
        """Screen one sample from its replicate scans."""
        r = self.predict_scans(X, wavelengths)
        frac_in = float(np.mean(r["in_domain"]))
        med = float(np.median(r["pred"]))
        q1, q3 = np.percentile(r["pred"], [25, 75])
        ident = float(np.median(r["r_blend"]))
        if frac_in < 0.5 or ident < 0.95:
            verdict = "OUT OF DOMAIN - not a paracetamol/talc blend like the calibration set"
        else:
            verdict = band(med)
        return dict(verdict=verdict, median_label_claim=med, iqr=(float(q1), float(q3)),
                    n_scans=len(r["pred"]), fraction_in_domain=frac_in, blend_similarity=ident,
                    api_similarity=float(np.nanmedian(r["r_api"])), scans=r)
