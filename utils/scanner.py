from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from utils.indicators import (
    rsi, stoch_k, sma, atr, pct_change,
    pattern_hammer, pattern_bullish_engulfing, rsi_divergence_proxy
)
from utils.earnings import is_earnings_soon
from utils.news import latest_news
from utils.market_regime import classify_regime

@dataclass
class Candidate:
    symbol: str
    close: float
    score: float
    score_breakdown: Dict[str, float]
    prob: str
    payoff: str
    label: str
    invalidation: str
    needs_t1: bool
    note: str
    data_status: str  # ok / insufficient
    earnings: str
    news: str

def fetch_ohlcv(symbol: str, days: int) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # standardize
    df = df.rename(columns={
        "Date":"Date","Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"
    })
    return df

def liquidity_ok(df: pd.DataFrame, min_vol: int, min_dvol: int) -> bool:
    if len(df) < 30:
        return False
    v = df["Volume"].tail(20).mean()
    px = df["Close"].tail(20).mean()
    dvol = v * px
    return (v >= min_vol) and (dvol >= min_dvol)

def compute_scores(df: pd.DataFrame, cfg: dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, object]]:
    # expects df with OHLCV
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    rsi_s = rsi(close, cfg["quant"]["rsi_period"])
    stoch_s = stoch_k(high, low, close, cfg["quant"]["stoch_period"])

    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    ma200 = sma(close, 200)
    atr14 = atr(high, low, close, 14)
    mom5 = pct_change(close, 5)

    last = df.iloc[-1]
    idx = len(df) - 1

    metrics = {
        "rsi": float(rsi_s.iloc[idx]) if not np.isnan(rsi_s.iloc[idx]) else np.nan,
        "stoch": float(stoch_s.iloc[idx]) if not np.isnan(stoch_s.iloc[idx]) else np.nan,
        "d_ma20": float((close.iloc[idx] / ma20.iloc[idx]) - 1) if not np.isnan(ma20.iloc[idx]) else np.nan,
        "d_ma50": float((close.iloc[idx] / ma50.iloc[idx]) - 1) if not np.isnan(ma50.iloc[idx]) else np.nan,
        "d_ma200": float((close.iloc[idx] / ma200.iloc[idx]) - 1) if not np.isnan(ma200.iloc[idx]) else np.nan,
        "mom5": float(mom5.iloc[idx]) if not np.isnan(mom5.iloc[idx]) else np.nan,
        "atr14": float(atr14.iloc[idx]) if not np.isnan(atr14.iloc[idx]) else np.nan
    }

    # Quant score (0..3)
    quant = 0.0
    # RSI oversold
    if not np.isnan(metrics["rsi"]) and metrics["rsi"] < cfg["quant"]["rsi_oversold"]:
        quant += 1.2
    # Stoch oversold
    if not np.isnan(metrics["stoch"]) and metrics["stoch"] < cfg["quant"]["stoch_oversold"]:
        quant += 0.8
    # below MA20/MA50 = pressure
    if not np.isnan(metrics["d_ma20"]) and metrics["d_ma20"] < -0.03:
        quant += 0.5
    if not np.isnan(metrics["d_ma50"]) and metrics["d_ma50"] < -0.05:
        quant += 0.5

    # Capitulation proxy: large range + vol spike
    if len(df) >= 30:
        rng = (df["High"] - df["Low"]).tail(5)
        rng_med = (df["High"] - df["Low"]).tail(30).median()
        vol5 = df["Volume"].tail(5).mean()
        vol30 = df["Volume"].tail(30).mean()
        if rng_med and vol30 and (rng.iloc[-1] > 1.5 * rng_med) and (vol5 > 1.2 * vol30):
            quant += 0.6

    quant = min(3.0, quant)

    # Pattern score (0..3)
    pattern = 0.0
    if pattern_hammer(df.tail(20)):
        pattern += 1.0
    if pattern_bullish_engulfing(df.tail(5)):
        pattern += 1.0
    if not np.isnan(metrics["rsi"]) and rsi_divergence_proxy(close, rsi_s):
        pattern += 1.0
    pattern = min(3.0, pattern)

    return {"quant": quant, "pattern": pattern}, metrics, {}

def fundamental_sanity(symbol: str) -> Tuple[float, str]:
    """
    Fundamentals (0..2). Only uses public yfinance fields.
    If missing core fields → return (0, "insufficient")
    """
    try:
        t = yf.Ticker(symbol)
        info = getattr(t, "info", None)
        if not info:
            return 0.0, "dados insuficientes (info vazio)"
        # Use a few broad red-flag checks; missing → insufficient
        mc = info.get("marketCap")
        de = info.get("debtToEquity")
        pm = info.get("profitMargins")
        if mc is None or pm is None:
            return 0.0, "dados insuficientes (marketCap/profitMargins)"
        score = 1.2  # baseline if not obviously broken
        # penalize extreme leverage if available
        if de is not None and isinstance(de, (int, float)) and de > 250:
            score -= 0.7
        # penalize strongly negative margins
        if isinstance(pm, (int, float)) and pm < -0.1:
            score -= 0.7
        score = max(0.0, min(2.0, score))
        return score, "ok"
    except Exception as e:
        return 0.0, f"dados insuficientes ({type(e).__name__})"

def narrative_score(symbol: str) -> Tuple[float, str]:
    title, pub, link = latest_news(symbol)
    if not title:
        return 0.0, "sem narrativa confirmável"
    # narrative is confirmation only: small points, no sentiment inference
    snippet = f"{title} ({pub})"
    if link:
        snippet += f"\n{link}"
    return 0.7, snippet

def classify_prob_payoff(total_score: float, quant: float, pattern: float) -> Tuple[str, str]:
    # coarse mapping
    if total_score >= 8.0 and quant >= 2.0 and pattern >= 1.5:
        return "alta", "médio"
    if total_score >= 7.0:
        return "média", "médio"
    if total_score >= 6.0:
        return "média", "baixo"
    return "baixa", "baixo"

def build_invalidation(df: pd.DataFrame) -> str:
    # simple: below last swing low (last 10 days low)
    low10 = df["Low"].tail(10).min()
    return f"Close < {low10:.2f}"

def is_extreme(metrics: Dict[str, float], cfg: dict) -> bool:
    r = metrics.get("rsi", np.nan)
    s = metrics.get("stoch", np.nan)
    return (not np.isnan(r) and r <= cfg["extreme_setup"]["rsi_extreme"]) or (not np.isnan(s) and s <= cfg["extreme_setup"]["stoch_extreme"])

def t_plus_1_needed(extreme: bool, pattern_score: float, cfg: dict) -> bool:
    if not cfg["t_plus_1"]["enabled"]:
        return False
    if extreme:
        return False
    # non-extreme: require confirmation unless pattern is strong
    return cfg["t_plus_1"]["non_extreme_requires_confirmation"] and pattern_score < 2.0

def scan_symbols(symbols: List[str], cfg: dict, region_name: str, audit) -> Tuple[List[Candidate], str]:
    regime, regime_note = classify_regime()
    if regime == "unknown":
        audit.add_insufficient(f"Regime: {regime_note}")
    audit.add_source("yfinance: OHLCV + VIX/SPY regime + earnings calendar + news")
    audit.ok("6_regime")

    candidates: List[Candidate] = []
    scanned = 0

    for sym in symbols[: cfg["outputs"]["max_candidates_scanned"]]:
        scanned += 1
        df = fetch_ohlcv(sym, cfg["quant"]["ma_windows"][-1] + 80)
        if df.empty or len(df) < 220:
            audit.add_insufficient(f"{sym}: OHLCV insuficiente")
            continue

        # liquidity gate
        if not liquidity_ok(df, cfg["liquidity"]["min_avg_volume_20d"], cfg["liquidity"]["min_avg_dollar_volume_20d"]):
            continue

        # earnings gate
        soon, ed = is_earnings_soon(sym, cfg["earnings"]["exclude_if_within_days"])
        if ed == "missing" and cfg["earnings"]["strict_if_missing"]:
            audit.add_insufficient(f"{sym}: earnings não confirmáveis → excluído")
            continue
        if soon:
            continue

        (subscores, metrics, _) = compute_scores(df, cfg)
        audit.ok("2_gate_oversold")

        # oversold gate (must be at least somewhat oversold)
        r = metrics.get("rsi", np.nan)
        st = metrics.get("stoch", np.nan)
        if np.isnan(r) or np.isnan(st):
            audit.add_insufficient(f"{sym}: RSI/Stoch insuficientes")
            continue
        if not (r < cfg["quant"]["rsi_oversold"] or st < cfg["quant"]["stoch_oversold"]):
            continue

        # pattern validation
        audit.ok("3_price_action")
        pattern_sc = subscores["pattern"]

        # fundamentals sanity (strictly: if insufficient, penalize heavily)
        f_sc, f_note = fundamental_sanity(sym)
        if "insuficientes" in f_note:
            # do not invent: penalize
            audit.add_insufficient(f"{sym}: fundamentais insuficientes → penalização")
        audit.ok("4_fundamentals")

        # narrative confirmation
        n_sc, n_note = narrative_score(sym)
        audit.ok("5_narrativa")

        # total score (0..10)
        w = cfg["scoring"]["weights"]
        total = (subscores["quant"] / 3.0) * w["quant"] \
              + (subscores["pattern"] / 3.0) * w["pattern"] \
              + (f_sc / 2.0) * w["fundamentals"] \
              + (n_sc / 2.0) * w["narrative"]

        total = float(round(total, 2))

        extreme = is_extreme(metrics, cfg)
        needs_t1 = t_plus_1_needed(extreme, subscores["pattern"], cfg)

        prob, payoff = classify_prob_payoff(total, subscores["quant"], subscores["pattern"])
        label = "Aceitável para Posse" if f_sc >= 1.0 else "Especulativa"

        inv = build_invalidation(df)
        close = float(df["Close"].iloc[-1])

        data_status = "ok"
        if f_sc == 0.0 and "insuficientes" in f_note:
            data_status = "insufficient"

        note = f"RSI={metrics['rsi']:.1f}, Stoch={metrics['stoch']:.1f}, Regime={regime}"

        candidates.append(Candidate(
            symbol=sym,
            close=round(close, 2),
            score=total,
            score_breakdown={
                "quant": subscores["quant"],
                "pattern": subscores["pattern"],
                "fund": round(f_sc, 2),
                "narr": round(n_sc, 2)
            },
            prob=prob,
            payoff=payoff,
            label=label,
            invalidation=inv,
            needs_t1=needs_t1,
            note=note,
            data_status=data_status,
            earnings=(ed if ed != "missing" else "missing"),
            news=n_note if n_note else "—"
        ))

    audit.ok("1_resets_ABC")
    return candidates, f"{region_name}: scanned={scanned}, kept={len(candidates)}"
