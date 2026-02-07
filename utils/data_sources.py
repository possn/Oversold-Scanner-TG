from __future__ import annotations

import datetime as dt
import pandas as pd
import yfinance as yf

def fetch_ohlcv(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days)
    df = yf.download(symbol, start=start.date(), end=end.date(), progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename_axis("Date").reset_index()
    # normalizar colunas
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

def fetch_next_earnings_date(symbol: str) -> dt.date | None:
    """
    Tenta obter earnings futuros via yfinance (sem key).
    Se falhar, devolve None (=> dados insuficientes).
    """
    try:
        t = yf.Ticker(symbol)
        # Em versões recentes:
        ed = t.get_earnings_dates(limit=8)
        if ed is None or ed.empty:
            return None
        # índice é datetime
        next_dt = ed.index.min()
        if next_dt is None:
            return None
        return next_dt.date()
    except Exception:
        return None

def fetch_basic_fundamentals(symbol: str) -> dict:
    """
    Fundamentos mínimos (sanity check): market cap, trailing PE, debt/equity (se existir).
    Se não existir, marca como None.
    """
    try:
        info = yf.Ticker(symbol).fast_info
        out = {}
        out["market_cap"] = getattr(info, "market_cap", None)
        out["last_price"] = getattr(info, "last_price", None)
        return out
    except Exception:
        return {"market_cap": None, "last_price": None}
