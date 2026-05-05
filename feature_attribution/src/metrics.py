import numpy as np


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rse(y_true, y_pred):
    numerator = np.sqrt(np.sum((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0:
        return np.nan
    return numerator / denominator


def rae(y_true, y_pred):
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    if denominator == 0:
        return np.nan
    return numerator / denominator


def corr(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return np.corrcoef(y_true, y_pred)[0, 1]


def evaluate_by_horizon(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true {y_true.shape}, pred {y_pred.shape}")

    if y_true.ndim != 3:
        raise ValueError("Expected shape (B, H, N)")

    H = y_true.shape[1]
    rows = []

    for h in range(H):
        yt = y_true[:, h, :]
        yp = y_pred[:, h, :]

        rows.append({
            "horizon": h + 1,
            "MAE": mae(yt, yp),
            "RMSE": rmse(yt, yp),
            "RSE": rse(yt, yp),
            "RAE": rae(yt, yp),
            "CORR": corr(yt, yp),
        })

    return rows