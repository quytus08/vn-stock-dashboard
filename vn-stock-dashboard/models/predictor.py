"""
predictor.py — Mô hình dự đoán xu hướng giá cổ phiếu.

Pipeline:
  1. Feature engineering: lag features, MA, RSI, momentum
  2. Train Linear Regression trên rolling window
  3. Dự đoán N ngày tiếp theo (recursive forecast)
  4. Tính confidence band ±1 std residual
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo feature matrix từ OHLCV."""
    feat = pd.DataFrame(index=df.index)

    feat["close"]    = df["close"]
    feat["ma5"]      = df["close"].rolling(5).mean()
    feat["ma10"]     = df["close"].rolling(10).mean()
    feat["ma20"]     = df["close"].rolling(20).mean()
    feat["rsi14"]    = _rsi(df["close"], 14)
    feat["mom5"]     = df["close"].pct_change(5)
    feat["mom10"]    = df["close"].pct_change(10)
    feat["vol_ratio"]= df["volume"] / df["volume"].rolling(20).mean()
    feat["hl_range"] = (df["high"] - df["low"]) / df["close"]

    for lag in [1, 2, 3, 5]:
        feat[f"lag_{lag}"] = df["close"].shift(lag)

    feat.dropna(inplace=True)
    return feat


def predict_trend(df: pd.DataFrame, n_days: int = 10) -> pd.DataFrame | None:
    """
    Dự đoán giá đóng cửa cho n_days ngày kế tiếp.

    Args:
        df:     DataFrame OHLCV (đã chuẩn hoá, index là DatetimeIndex)
        n_days: Số ngày muốn dự đoán

    Returns:
        DataFrame với cột 'predicted', 'upper', 'lower' và index là ngày tương lai.
        None nếu không đủ dữ liệu.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    feat = _build_features(df)
    if len(feat) < 40:
        return None

    # Target: giá ngày mai
    X = feat.drop(columns=["close"]).values
    y = feat["close"].values

    # Dùng tối đa 300 điểm gần nhất để train
    window = min(len(X) - 1, 300)
    X_train = X[-window-1:-1]
    y_train = y[-window:]

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_s, y_train)

    # Residuals để tính confidence band
    y_pred_train = model.predict(X_s)
    residual_std = np.std(y_train - y_pred_train)

    # Recursive forecast
    last_df    = df.copy()
    pred_prices = []
    last_date   = df.index[-1]

    for _ in range(n_days):
        feat_curr = _build_features(last_df)
        if feat_curr.empty:
            break
        x_new = feat_curr.drop(columns=["close"]).values[-1].reshape(1, -1)
        x_new_s = scaler.transform(x_new)
        p = model.predict(x_new_s)[0]
        pred_prices.append(p)

        # Thêm ngày dự đoán vào df để predict ngày tiếp theo
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:   # bỏ qua T7, CN
            next_date += timedelta(days=1)
        last_date = next_date

        new_row = pd.DataFrame({
            "open":   [p * 0.999],
            "high":   [p * 1.005],
            "low":    [p * 0.995],
            "close":  [p],
            "volume": [last_df["volume"].rolling(5).mean().iloc[-1]],
        }, index=[next_date])
        last_df = pd.concat([last_df, new_row])

    if not pred_prices:
        return None

    # Tạo index ngày tương lai (bỏ T7/CN)
    future_dates = []
    d = df.index[-1]
    while len(future_dates) < len(pred_prices):
        d += timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d)

    result = pd.DataFrame({
        "predicted": pred_prices,
        "upper":     [p + residual_std * 1.2 for p in pred_prices],
        "lower":     [p - residual_std * 1.2 for p in pred_prices],
    }, index=future_dates)

    return result
