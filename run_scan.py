from __future__ import annotations

import os
import json
import math
import datetime as dt
import pandas as pd

from utils.telegram import send_telegram_message
from utils.universe import (
    get_sp500_symbols, get_nasdaq100_symbols,
    get_psi20_symbols, get_euronext100_symbols
)
from utils.data_sources import fetch_ohlcv, fetch_next_earnings_date, fetch_basic_fundamentals
from utils.indicators import rsi, stochastic_k, sma, roc, zscore, candle_reversal_score

def pct(a: float) -> str:
    return f"{a:.1f}%"

def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def compute_features(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["RSI14"] = rsi(df["Close"], 14)
    df["STOCH14"] = stochastic_k(df, 14)
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)
    df["ROC20"] = roc(df["Close"], 20)
    df["VOL_Z20"] = zscore(df["Volume"], 20)

    last = df.iloc[-1]
    feats = {
        "close": float(last["Close"]),
        "rsi14": float(last["RSI14"]),
        "stoch14": float(last["STOCH14"]),
        "dist_sma20_pct": float((last["Close"] / last["SMA20"] - 1) * 100) if not math.isnan(last["SMA20"]) else None,
        "dist_sma50_pct": float((last["Close"] / last["SMA50"] - 1) * 100) if not math.isnan(last["SMA50"]) else None,
        "dist_sma200_pct": float((last["Close"] / last["SMA200"] - 1) * 100) if not math.isnan(last["SMA200"]) else None,
        "roc20": float(last["ROC20"]) if not math.isnan(last["ROC20"]) else None,
        "vol_z20": float(last["VOL_Z20"]) if not math.isnan(last["VOL_Z20"]) else None,
        "pattern_raw": float(candle_reversal_score(df[["Open","High","Low","Close"]].tail(5))),
        "avg_dollar_vol_20d": float((df["Close"].tail(20) * df["Volume"].tail(20)).mean()) if len(df) >= 20 else 0.0,
        "last_date": str(df.iloc[-1]["Date"].date())
    }
    return feats

def oversold_gate(feats: dict, cfg: dict) -> tuple[bool, list[str]]:
    g = cfg["oversold_gate"]
    reasons = []

    def check(cond, msg):
        if not cond:
            reasons.append(msg)

    check(feats["rsi14"] <= g["rsi14_max"], f"RSI14>{g['rsi14_max']}")
    check(feats["stoch14"] <= g["stoch14_max"], f"Stoch14>{g['stoch14_max']}")
    check(feats["dist_sma20_pct"] is not None and feats["dist_sma20_pct"] <= g["dist_sma20_min_pct"], "dist>SMA20 insuf/alto")
    check(feats["dist_sma50_pct"] is not None and feats["dist_sma50_pct"] <= g["dist_sma50_min_pct"], "dist>SMA50 insuf/alto")
    check(feats["dist_sma200_pct"] is not None and feats["dist_sma200_pct"] <= g["dist_sma200_min_pct"], "dist>SMA200 insuf/alto")
    check(feats["roc20"] is not None and feats["roc20"] <= g["roc20_max"], "ROC20 insuf/alto")
    check(feats["vol_z20"] is not None and feats["vol_z20"] >= g["volume_capitulation_z_min"], "sem capitulação (VOL_Z20 baixo)")
    return (len(reasons) == 0), reasons

def earnings_filter(symbol: str, cfg: dict) -> tuple[bool, str]:
    days = cfg["earnings"]["exclude_if_earnings_within_days"]
    exclude_if_none = cfg["earnings"]["exclude_if_cannot_confirm"]

    d = fetch_next_earnings_date(symbol)
    if d is None:
        return (not exclude_if_none), "dados insuficientes (earnings não confirmáveis)"
    today = dt.date.today()
    delta = (d - today).days
    if delta < 0:
        return True, f"earnings já ocorreram ({d})"
    if delta <= days:
        return False, f"excluída: earnings em {delta}d ({d})"
    return True, f"earnings ok ({d}, {delta}d)"

def quant_score(feats: dict, cfg: dict) -> float:
    """
    0..3 (aprox). Mais oversold => maior.
    """
    s = 0.0
    # RSI: 32->0, 20->1
    s += max(0.0, min(1.0, (cfg["oversold_gate"]["rsi14_max"] - feats["rsi14"]) / 12.0))
    # Stoch: 20->0, 5->1
    s += max(0.0, min(1.0, (cfg["oversold_gate"]["stoch14_max"] - feats["stoch14"]) / 15.0))
    # Distâncias agregadas: mais abaixo das MMs -> mais score
    dists = [feats["dist_sma20_pct"], feats["dist_sma50_pct"], feats["dist_sma200_pct"]]
    dists = [d for d in dists if d is not None]
    if dists:
        s += max(0.0, min(1.0, (-sum(dists)/len(dists)) / 10.0))
    return min(3.0, s * 1.0)

def pattern_score(feats: dict) -> float:
    # pattern_raw 0..1.5 -> map 0..3 (cap)
    return min(3.0, feats["pattern_raw"] * 2.0)

def fundamentals_score(fund: dict) -> tuple[float, str]:
    """
    Sanity check minimalista:
    - se não há market cap => dados insuficientes => 0.5 (penalizado mas não mata)
    - se market cap muito baixo => penaliza (pode ser especulativo/ilíquido)
    """
    mc = fund.get("market_cap")
    if mc is None:
        return 0.5, "dados insuficientes (market cap)"
    if mc < 2e9:
        return 0.5, "micro/small cap (sanity)"
    if mc < 10e9:
        return 1.2, "mid cap (ok)"
    return 2.0, "large cap (ok)"

def narrative_stub(symbol: str) -> tuple[float, str]:
    """
    Sem APIs pagas de news: aqui é um proxy conservador:
    - devolve 0.5 fixo com nota 'sem retrieve de notícia' (para não inventar).
    Se quiseres, acrescentamos RSS públicos (Reuters/FT/PR) com parsing e citações.
    """
    return 0.5, "narrativa: proxy (sem retrieve de notícia no modo base)"

def probability_payoff_labels(total_score: float, feats: dict) -> tuple[str, str, str]:
    """
    Heurística simples e explícita:
    - Probabilidade sobe com score e com padrões de reversão.
    - Payoff sobe com distância às MMs (maior snapback potencial) mas risco também.
    """
    prob = "média"
    if total_score >= 8.0 and feats["pattern_raw"] >= 0.7:
        prob = "alta"
    elif total_score < 6.8:
        prob = "baixa"

    avg_dist = 0.0
    cnt = 0
    for k in ["dist_sma20_pct","dist_sma50_pct","dist_sma200_pct"]:
        if feats[k] is not None:
            avg_dist += feats[k]; cnt += 1
    avg_dist = avg_dist / cnt if cnt else 0.0

    payoff = "médio"
    if avg_dist <= -12:
        payoff = "alto"
    elif avg_dist > -6:
        payoff = "baixo"

    asym = "assimetria moderada"
    if payoff == "alto" and prob in ["média","baixa"]:
        asym = "alta assimetria (payoff alto / prob <= média)"
    elif prob == "alta" and payoff in ["baixo","médio"]:
        asym = "alta probabilidade (payoff contido)"

    return prob, payoff, asym

def run_region(region: str) -> dict:
    cfg = load_config()

    if region == "US":
        universe = list(set(get_sp500_symbols() + get_nasdaq100_symbols()))
        label = "EUA (S&P500 + Nasdaq100)"
    elif region == "EU":
        universe = list(set(get_euronext100_symbols() + get_psi20_symbols()))
        label = "Europa (Euronext100 + PSI20)"
    else:
        raise ValueError("region must be US or EU")

    # 3 resets independentes A/B/C (shuffle determinístico diferente)
    base = sorted(universe)
    A = base[::3]
    B = base[1::3]
    C = base[2::3]
    resets = {"A": A, "B": B, "C": C}

    results = {k: [] for k in resets}
    audit = {"label": label, "timestamp_utc": dt.datetime.utcnow().isoformat(), "sources": ["Wikipedia (constituintes)", "Yahoo Finance via yfinance (OHLCV + earnings quando disponível)"], "excluded": []}

    for rk, syms in resets.items():
        candidates = []
        for s in syms:
            df = fetch_ohlcv(s, lookback_days=400)
            if df.empty or len(df) < 220:
                audit["excluded"].append((s, "dados insuficientes (OHLCV histórico)"))
                continue

            feats = compute_features(df)

            # liquidez
            if feats["avg_dollar_vol_20d"] < cfg["liquidity"]["min_avg_dollar_volume_20d"]:
                audit["excluded"].append((s, "ilíquida (avg $vol 20d)"))
                continue

            # earnings
            ok_earn, earn_msg = earnings_filter(s, cfg)
            if not ok_earn:
                audit["excluded"].append((s, earn_msg))
                continue
            if "dados insuficientes" in earn_msg and cfg["earnings"]["exclude_if_cannot_confirm"]:
                audit["excluded"].append((s, earn_msg))
                continue

            # gate oversold
            ok_gate, gate_reasons = oversold_gate(feats, cfg)
            if not ok_gate:
                # penaliza: não entra
                continue

            fund = fetch_basic_fundamentals(s)
            f_score, f_note = fundamentals_score(fund)

            q = quant_score(feats, cfg)
            p = pattern_score(feats)
            n_score, n_note = narrative_stub(s)

            total = (q/3)*cfg["scoring"]["weights"]["quant"] + (p/3)*cfg["scoring"]["weights"]["pattern"] + (f_score/2)*cfg["scoring"]["weights"]["fundamentals"] + (n_score/2)*cfg["scoring"]["weights"]["narrative"]
            # total já está em escala 0..10 aprox
            if total < cfg["scoring"]["min_total_score_to_consider"]:
                continue

            prob, payoff, asym = probability_payoff_labels(total, feats)

            # T+1
            executavel = True
            if cfg["tplus1"]["enable"] and feats["rsi14"] > cfg["tplus1"]["extreme_rsi14_max"]:
                executavel = False

            candidates.append({
                "symbol": s,
                "score": round(float(total), 2),
                "rsi14": round(feats["rsi14"], 1),
                "stoch14": round(feats["stoch14"], 1),
                "dist20": round(feats["dist_sma20_pct"], 1),
                "dist50": round(feats["dist_sma50_pct"], 1),
                "dist200": round(feats["dist_sma200_pct"], 1),
                "roc20": round(feats["roc20"], 1),
                "volz": round(feats["vol_z20"], 2),
                "executavel": executavel,
                "fund_note": f_note,
                "narrative": n_note,
                "prob": prob,
                "payoff": payoff,
                "asym": asym,
                "last_date": feats["last_date"]
            })

        # ranking por score e “grau de oversold” implícito
        candidates = sorted(candidates, key=lambda x: (x["score"], -x["rsi14"]), reverse=True)
        results[rk] = candidates[:cfg["selection"]["top_per_reset"]]

    # intersecção >=2/3
    from collections import Counter
    all_syms = [x["symbol"] for k in results for x in results[k]]
    counts = Counter(all_syms)
    inter = [s for s, c in counts.items() if c >= cfg["selection"]["intersection_min_resets"]]

    # escolher executáveis e manter diversidade prob/payoff
    inter_rows = []
    for rk in results:
        for row in results[rk]:
            if row["symbol"] in inter:
                inter_rows.append(row)
    # deduplicar por melhor score
    best = {}
    for r in inter_rows:
        if (r["symbol"] not in best) or (r["score"] > best[r["symbol"]]["score"]):
            best[r["symbol"]] = r
    inter_best = sorted(best.values(), key=lambda x: x["score"], reverse=True)

    return {"audit": audit, "results": results, "intersection": inter_best}

def format_message(payload: dict, region: str) -> str:
    audit = payload["audit"]
    results = payload["results"]
    inter = payload["intersection"]

    def fmt_row(r):
        tag = "EXEC" if r["executavel"] else "T+1"
        # rótulo posse vs especulativa (heurística simples)
        aceitavel = "Aceitável para posse" if (r["prob"] == "alta" and r["payoff"] in ["baixo","médio"]) else "Especulativa"
        inval = f"invalidação: abaixo do mínimo recente / fecho abaixo do último mínimo (manual)"
        return (f"<b>{r['symbol']}</b> [{tag}] — Score {r['score']}/10 | "
                f"P:{r['prob']} × Payoff:{r['payoff']} | {aceitavel}\n"
                f"RSI {r['rsi14']} | Stoch {r['stoch14']} | d20 {r['dist20']}% d50 {r['dist50']}% d200 {r['dist200']}% | ROC20 {r['roc20']}% | VOLz {r['volz']}\n"
                f"{r['asym']}\n"
                f"{r['narrative']}\n"
                f"{inval}\n")

    lines = []
    lines.append(f"<b>OVERSOLD SCANNER — {audit['label']}</b>")
    lines.append(f"UTC: {audit['timestamp_utc']}")
    lines.append(f"Fontes: " + " | ".join(audit["sources"]))
    lines.append("")
    lines.append("<b>AUDITORIA (alto nível)</b>")
    lines.append("✓ Universos via Wikipedia (constituintes)")
    lines.append("✓ OHLCV + volume via Yahoo (yfinance)")
    lines.append("✓ Earnings: yfinance (se falhar => dados insuficientes => exclui)")
    lines.append("✓ Gate multifatorial + padrões simples + sanity fundamental + narrativa proxy")
    lines.append("✓ 3 resets A/B/C + intersecção ≥2/3")
    lines.append("")

    lines.append("<b>TOP 5 por Reset</b>")
    for k in ["A","B","C"]:
        lines.append(f"\n<b>Reset {k}</b>")
        if not results[k]:
            lines.append("— (sem candidatos suficientes)")
        else:
            for r in results[k]:
                lines.append(f"- {r['symbol']} (Score {r['score']}, RSI {r['rsi14']}, {('EXEC' if r['executavel'] else 'T+1')})")

    lines.append("\n<b>INTERSEÇÃO (≥2/3) — executáveis vs T+1</b>")
    if not inter:
        lines.append("— Nenhuma ação passou a intersecção com dados suficientes.")
    else:
        for r in inter[:8]:  # limite de tamanho
            lines.append("\n" + fmt_row(r))

    # Nota de controlo: repetição de nomes é normal em regimes de queda
    lines.append("\n<b>Nota</b>: repetição de nomes pode ocorrer se o oversold persistir; o filtro de earnings+liquidez impede lixo/ruído.")
    return "\n".join(lines)

def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    region = os.environ.get("REGION", "US").upper()

    payload = run_region(region)
    msg = format_message(payload, region)

    send_telegram_message(token, chat_id, msg)

if __name__ == "__main__":
    main()
