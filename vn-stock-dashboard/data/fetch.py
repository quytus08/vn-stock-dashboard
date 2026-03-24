"""
fetch.py — Lấy dữ liệu chứng khoán Việt Nam qua thư viện vnstock.
Hỗ trợ fallback sang dữ liệu mẫu nếu chưa cài vnstock hoặc mất mạng.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_popular_tickers():
    """Danh sách mã cổ phiếu phổ biến trên HOSE/HNX."""
    return [
        "VNM", "VCB", "BID", "CTG", "TCB",
        "VIC", "VHM", "HPG", "MWG", "FPT",
        "MSN", "GAS", "SAB", "VPB", "ACB",
        "STB", "MBB", "SSI", "HDB", "REE",
    ]


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Lấy dữ liệu OHLCV cho một mã cổ phiếu.

    Thử theo thứ tự:
      1. vnstock (VNDIRECT source)
      2. vnstock (SSI source)
      3. Dữ liệu giả lập (demo mode) — để chạy offline

    Returns:
        DataFrame với index là DatetimeIndex và các cột:
        open, high, low, close, volume
    """
    # ── Attempt 1: vnstock ────────────────────────────────────────────────────
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=ticker, source="VCI")
        df = stock.quote.history(
            start=start_date,
            end=end_date,
            interval="1D",
        )
        if df is not None and not df.empty:
            df = _normalize(df)
            if df is not None:
                return df
    except Exception:
        pass

    # ── Attempt 2: vnstock SSI fallback ──────────────────────────────────────
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=ticker, source="TCBS")
        df = stock.quote.history(
            start=start_date,
            end=end_date,
            interval="1D",
        )
        if df is not None and not df.empty:
            df = _normalize(df)
            if df is not None:
                return df
    except Exception:
        pass

    # ── Attempt 3: Demo / offline fallback ───────────────────────────────────
    return _generate_demo_data(ticker, start_date, end_date)


def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """Chuẩn hoá tên cột và kiểu dữ liệu về định dạng chung."""
    df = df.copy()

    # Chuẩn hoá tên cột về chữ thường
    df.columns = [c.lower().strip() for c in df.columns]

    # Map tên cột phổ biến
    rename_map = {
        "time": "date", "tradingdate": "date",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    }
    df.rename(columns=rename_map, inplace=True)

    # Đảm bảo đủ cột
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    # Set index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


def _generate_demo_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Sinh dữ liệu ngẫu nhiên thực tế để demo khi không có internet.
    Dùng random walk với drift nhẹ, volatility thực tế (~1.5%/ngày).
    """
    np.random.seed(abs(hash(ticker)) % (2**31))

    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    if n == 0:
        dates = pd.bdate_range(end=end_date, periods=120)
        n = 120

    # Giá khởi đầu thực tế theo mã
    base_prices = {
        "VNM": 75000, "VCB": 90000, "BID": 45000, "TCB": 35000,
        "HPG": 28000, "FPT": 95000, "MWG": 55000, "VIC": 48000,
        "VHM": 42000, "MSN": 68000,
    }
    p0 = base_prices.get(ticker, 50000)

    returns  = np.random.normal(0.0003, 0.015, n)
    closes   = p0 * np.cumprod(1 + returns)
    opens    = closes * np.random.uniform(0.99, 1.01, n)
    highs    = np.maximum(opens, closes) * np.random.uniform(1.002, 1.015, n)
    lows     = np.minimum(opens, closes) * np.random.uniform(0.985, 0.998, n)
    volumes  = np.random.randint(500_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({
        "open":   opens.round(0),
        "high":   highs.round(0),
        "low":    lows.round(0),
        "close":  closes.round(0),
        "volume": volumes,
    }, index=dates)

    return df
