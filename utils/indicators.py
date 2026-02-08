import numpy as np
import pandas as pd

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    lowest = low.rolling(period).min()
    highest = high.rolling(period).max()
    return 100 * (close - lowest) / (highest - lowest)

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def pct_change(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n)

def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    # expects columns: Open High Low Close Volume
    out = df.copy()
    out["body"] = (out["Close"] - out["Open"]).abs()
    out["range"] = (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick"] = out["High"] - out[["Open", "Close"]].max(axis=1)
    out["lower_wick"] = out[["Open", "Close"]].min(axis=1) - out["Low"]
    out["is_green"] = out["Close"] > out["Open"]
    return out

def pattern_hammer(df: pd.DataFrame) -> bool:
    # last candle hammer-ish: long lower wick, small body
    if len(df) < 2:
        return False
    c = candle_features(df).iloc[-1]
    if pd.isna(c["range"]):
        return False
    return (c["lower_wick"] >= 2.0 * c["body"]) and (c["upper_wick"] <= 0.6 * c["body"])

def pattern_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    return (prev["Close"] < prev["Open"]) and (last["Close"] > last["Open"]) and (last["Open"] < prev["Close"]) and (last["Close"] > prev["Open"])

def rsi_divergence_proxy(close: pd.Series, rsi_series: pd.Series) -> bool:
    # proxy: price makes lower low last 10 bars but RSI makes higher low
    if len(close) < 20:
        return False
    w = 10
    c = close.iloc[-w:]
    r = rsi_series.iloc[-w:]
    if c.isna().any() or r.isna().any():
        return False
    price_ll = c.iloc[-1] < c.min()
    rsi_hl = r.iloc[-1] > r.min()
    return bool(price_ll and rsi_hl)
