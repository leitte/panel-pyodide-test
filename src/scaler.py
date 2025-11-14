import pandas as pd
import numpy as np
import json

from typing import Dict, List, Tuple

# ----------------------------
# Scaler: robust (train-only) for inputs; percentile MinMax for target Δ
# ----------------------------
class RobustInputScaler:
    def __init__(self, winsor_q=(0.001,0.999)):
        self.q = winsor_q
        self.median_: Dict[str,float] = {}
        self.iqr_: Dict[str,float] = {}
        self.clip_: Dict[str,Tuple[float,float]] = {}
        self.col_names: List[str] = []

    def save(self, path: str):
        payload = {
            "winsor_q": self.q,
            "median_": self.median_,
            "iqr_": self.iqr_,
            "clip_": self.clip_,
            "col_names": self.col_names,
        }
        with open(path, "w") as f: 
            json.dump(payload, f)

    def load(self, path: str):
        with open(path) as f: 
            d = json.load(f)

        self.q = tuple(d["winsor_q"]) if d["winsor_q"] else None
        self.median_ = {k: float(v) for k, v in d["median_"].items()}
        self.iqr_    = {k: float(v) for k, v in d["iqr_"].items()}
        self.clip_   = {k: (float(v[0]), float(v[1])) for k, v in d["clip_"].items()}
        self.col_names = d["col_names"]

    def fit(self, dfs: List[pd.DataFrame], cols: List[str]):
        self.col_names = cols
        cat = pd.concat([df[cols] for df in dfs], axis=0)
        q_lo = cat.quantile(self.q[0])
        q_hi = cat.quantile(self.q[1])
        cat = cat.clip(lower=q_lo, upper=q_hi, axis=1)
        med = cat.median()
        iqr = cat.quantile(0.75) - cat.quantile(0.25)
        eps = 1e-6
        for c in cols:
            self.median_[c] = float(med[c])
            self.iqr_[c] = float(max(iqr[c], eps))
            self.clip_[c] = (float(q_lo[c]), float(q_hi[c]))

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            # use this for clipping
            # lo, hi = self.clip_[c]
            # x = out[c].clip(lo, hi)
            x = out[c]
            if self.iqr_[c] > 1e-5:
                out[c] = (x - self.median_[c]) / self.iqr_[c]
            else:
                out[c] = (x - self.median_[c])
        return out
    
    def inverse(self, X, cols=None):
        """
        Reverse the robust scaling.

        X : pd.DataFrame or np.ndarray
            If ndarray, assumes column order = self.cols.
        cols : list of str, optional
            If given, restrict inverse to these columns.
        """

        if (cols is None) and (X.shape[1] == len(self.col_names)):
            cols = self.col_names
        
        if len(cols) != X.shape[1]:
            raise ValueError(f'Inverse scaling of the input does not work. No column names are given and the numbers do not match.')

        if isinstance(X, np.ndarray):
            # Assume column order = self.cols
            X = np.asarray(X, dtype=np.float32)
            out = np.empty_like(X)
            for j, c in enumerate(cols):
                if self.iqr_[c] > 1e-5:
                    out[:, j] = X[:, j] * self.iqr_[c] + self.median_[c]
                else:
                    out[:, j] = X[:, j] + self.median_[c]
            return out
        else:
            raise TypeError("X must be DataFrame or ndarray")
    
class RobustTargetScaler:
    """
    Per-target robust scaler with winsorization, for arrays shaped:
      - [N, H, K]  or [N, K]  (K = number of targets)
    Fits per target across all samples and horizons.
    """
    def __init__(self, winsor_q: Tuple[float,float]=(0.01, 0.99), target_names: List[str] | None = None):
        self.q = winsor_q
        self.target_names = target_names  # optional, for bookkeeping
        self.lo_: Dict[int,float] = {}
        self.hi_: Dict[int,float] = {}
        self.med_: Dict[int,float] = {}
        self.iqr_: Dict[int,float] = {}

    def save(self, path: str):
        payload = {
            "winsor_q": self.q,
            "target_names": self.target_names,
            "low": self.lo_,
            "high": self.hi_,
            "median": self.med_,
            "iqr_": self.iqr_,
        }
        with open(path, "w") as f: 
            json.dump(payload, f)

    def load(self, path: str):
        with open(path) as f: 
            d = json.load(f)

        self.q = tuple(d["winsor_q"]) if d["winsor_q"] else None
        self.lo_ = {int(k): float(v) for k, v in d["low"].items()}
        self.hi_ = {int(k): float(v) for k, v in d["high"].items()}
        self.med_ = {int(k): float(v) for k, v in d["median"].items()}
        self.iqr_    = {int(k): float(v) for k, v in d["iqr_"].items()}

    def _flatten_target(self, y: np.ndarray, k: int) -> np.ndarray:
        # y: [N,H,K] or [N,K] -> 1D over all samples/horizons for target k
        if y.ndim == 3:   # [N,H,K]
            return y[..., k].reshape(-1)
        elif y.ndim == 2: # [N,K]
            return y[:, k].reshape(-1)
        else:
            raise ValueError(f"Expected y.ndim in {{2,3}}, got {y.ndim}")

    def fit(self, y: np.ndarray):
        K = y.shape[-1]
        eps = 1e-6
        for k in range(K):
            z = self._flatten_target(y, k)
            lo = np.quantile(z, self.q[0])
            hi = np.quantile(z, self.q[1])
            z = np.clip(z, lo, hi)
            med = np.median(z)
            q75, q25 = np.percentile(z, 75), np.percentile(z, 25)
            iqr = max(q75 - q25, eps)
            self.lo_[k], self.hi_[k], self.med_[k], self.iqr_[k] = float(lo), float(hi), float(med), float(iqr)

    def transform(self, y: np.ndarray) -> np.ndarray:
        y2 = y.astype(np.float32, copy=True)
        K = y2.shape[-1]
        for k in range(K):
            lo, hi = self.lo_[k], self.hi_[k]
            med, iqr = self.med_[k], self.iqr_[k]
            #yk = np.clip(y2[..., k], lo, hi)
            yk = y2[..., k]
            y2[..., k] = (yk - med) / iqr
        return y2

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        y = y_scaled.astype(np.float32, copy=True)
        K = y.shape[-1]
        for k in range(K):
            med, iqr = self.med_[k], self.iqr_[k]
            y[..., k] = y[..., k] * iqr + med
        return y
