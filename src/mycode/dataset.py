import pandas as pd
import numpy as np

from mycode.scaler import RobustInputScaler, RobustTargetScaler

class WindowedDeltaDataset:
    """
    Windowed dataset for multi-target forecasting with delta outputs.

    Shapes:
      X      : [N, window, F]     (inputs, scaled)
      y      : [N, horizon, K]    (deltas of targets, scaled)
      y_last : [N, K]             (last true target level at t, unscaled)
      ctx    : [N, Cctx]          (optional static context per window, scaled/safe)

    Usage:
      - During training, predict `y` (scaled deltas).
      - For reconstruction: `y_pred_levels = y_last + cumsum(delta_scaler.inverse(y_pred), axis=1)`
      - If `context_cols` is provided, batches return (X, y, y_last, ctx); else (X, y, y_last).
    """
    def __init__(self, 
                 df: pd.DataFrame, 
                 input_scaler: RobustInputScaler,
                 delta_scaler: RobustTargetScaler,
                ):
        self.input_scaler = input_scaler
        self.target_scaler = delta_scaler

        self.X, self.y, self.y_last, self.index = [], [], [], []
        self.ctx = []

        W = 140
        H = 30
        stride = 10

        self.window = W
        self.horizon = H
        self.stride = stride

        in_features = input_scaler.col_names
        target_features = ['PY23', 'PDI701', 'PDI702', 'T705', 'T709', 'T711', 'T712', 'T701', 'T702', 'T703', 'T704', 'FT703', 'FT704']
        self.target_scaler.target_names = target_features

        # --- scale inputs ---
        df_in = input_scaler.transform(df, in_features)
        X_all = df_in[in_features].to_numpy(dtype=np.float32)

        # --- target levels (unscaled) ---
        t = df[target_features].to_numpy(dtype=np.float32)  # [T,K]
        n = len(t)

        # --- windowing ---
        starts = list(range(0, n - W - H + 1, stride))

        X_win = np.stack([X_all[s : s + W, :] for s in starts], axis=0)   # [N, W, F]
        
        deltas = np.stack([
            t[s+W : s+W+H, :] - t[s+W-1 : s+W+H-1, :]   # shape [H,K]
            for s in starts
        ], axis=0).astype(np.float32)  # [N, H, K]

        # --- scale deltas if scaler is provided ---
        if delta_scaler is not None:
            y_win = delta_scaler.transform(deltas)
        else:
            y_win = deltas

        # y_win = np.stack([y_scaled[i, :, :] for i, _ in enumerate(starts)], axis=0)     # [N, H, K]
        y_last_vals = np.array([t[s + W - 1, :] for s in starts], dtype=np.float32)  # [N,K]

        self.X.append(X_win)
        self.y.append(y_win)
        self.y_last.append(y_last_vals)



        if len(self.X) == 0:
            raise ValueError("No windows created. Check window/horizon/stride and data length.")

        # concatenate
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        self.y_last = np.concatenate(self.y_last, axis=0)

    def __len__(self): 
        return self.X.shape[0]