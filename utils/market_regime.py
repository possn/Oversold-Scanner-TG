import yfinance as yf
import pandas as pd
from utils.indicators import sma

def fetch_series(symbol: str, days: int = 260) -> pd.Series:
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df["Close"]

def classify_regime() -> (str, str):
    """
    Simple regime:
    - risk_off: SPY < MA50 AND VIX > 20
    - risk_on:  SPY > MA50 AND VIX < 18
    - neutral otherwise
    """
    spy = fetch_series("SPY")
    vix = fetch_series("^VIX")
    if spy.empty or vix.empty or len(spy) < 60 or len(vix) < 10:
        return "unknown", "dados insuficientes (SPY/VIX)"

    spy_ma50 = sma(spy, 50).iloc[-1]
    spy_last = spy.iloc[-1]
    vix_last = vix.iloc[-1]

    if spy_last < spy_ma50 and vix_last > 20:
        return "risk_off", f"SPY<{spy_ma50:.2f}, VIX={vix_last:.1f}"
    if spy_last > spy_ma50 and vix_last < 18:
        return "risk_on", f"SPY>{spy_ma50:.2f}, VIX={vix_last:.1f}"
    return "neutral", f"SPY={spy_last:.2f} vs MA50={spy_ma50:.2f}, VIX={vix_last:.1f}"
