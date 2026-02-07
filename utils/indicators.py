import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_ = 100 - (100 / (1 + rs))
    return rsi_.fillna(method="bfill")

def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    return k.replace([np.inf, -np.inf], np.nan).fillna(method="bfill")

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def roc(series: pd.Series, period: int = 20) -> pd.Series:
    return 100 * (series / series.shift(period) - 1.0)

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)

def candle_reversal_score(df: pd.DataFrame) -> float:
    """
    Heurística simples (pública e verificável) de reversão:
    - martelo/hammer aproximado
    - engulfing bullish aproximado
    Devolve 0..1.5 (para ser mapeado no score pattern).
    """
    if len(df) < 3:
        return 0.0
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last["Close"] - last["Open"])
    rng = last["High"] - last["Low"]
    if rng == 0:
        return 0.0

    lower_wick = min(last["Open"], last["Close"]) - last["Low"]
    upper_wick = last["High"] - max(last["Open"], last["Close"])

    hammer = (lower_wick / rng > 0.55) and (upper_wick / rng < 0.2) and (body / rng < 0.25)
    engulf = (last["Close"] > last["Open"]) and (prev["Close"] < prev["Open"]) and (last["Close"] >= prev["Open"]) and (last["Open"] <= prev["Close"])

    score = 0.0
    if hammer:
        score += 0.8
    if engulf:
        score += 0.7
    return score
