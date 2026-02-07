from __future__ import annotations

import os
import json
import datetime as dt
import pandas as pd

from utils.telegram import send_telegram_message

HISTORY_PATH = "data/history.csv"
CONFIG_PATH = "config.json"

def load_history() -> pd.DataFrame:
    try:
        df = pd.read_csv(HISTORY_PATH)
        return df
    except Exception:
        return pd.DataFrame()

def save_history(df: pd.DataFrame) -> None:
    os.makedirs("data", exist_ok=True)
    df.to_csv(HISTORY_PATH, index=False)

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def evaluate_week(df: pd.DataFrame) -> dict:
    """
    Espera que history tenha colunas mínimas:
    date, region, symbol, entry_close, max_3d, max_5d, min_5d
    """
    if df.empty:
        return {"ok": False, "reason": "sem histórico"}

    # última semana (7 dias)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=7)
    w = df[df["date"] >= cutoff].copy()
    if w.empty:
        return {"ok": False, "reason": "sem dados na última semana"}

    # sucesso: max_3d >= +2% (ajustável)
    w["ret_3d"] = (w["max_3d"] / w["entry_close"] - 1.0) * 100
    w["dd_5d"] = (w["min_5d"] / w["entry_close"] - 1.0) * 100
    w["success"] = w["ret_3d"] >= 2.0

    out = {
        "ok": True,
        "n": int(len(w)),
        "hit_rate": float(w["success"].mean() * 100),
        "avg_rebound_3d": float(w["ret_3d"].mean()),
        "avg_dd_5d": float(w["dd_5d"].mean()),
        "best_rebound": float(w["ret_3d"].max()),
        "worst_dd": float(w["dd_5d"].min())
    }
    return out

def propose_small_adjustments(cfg: dict, metrics: dict) -> tuple[dict, str]:
    """
    Ajuste automático CONSERVADOR:
    - só mexe se houver >=30 observações e melhoria esperada plausível.
    Aqui fazemos uma regra simples:
      se hit_rate < 60% e dd muito negativo => apertar gate (rsi_max desce 2)
      se hit_rate > 75% e pouco sinal => relaxar ligeiro (rsi_max sobe 1)
    Isto é deliberadamente simples para evitar overfitting.
    """
    note = "Sem alterações automáticas (evidência insuficiente ou risco de overfitting)."
    n = metrics.get("n", 0)
    if not metrics.get("ok") or n < 30:
        return cfg, note

    hit = metrics["hit_rate"]
    dd = metrics["avg_dd_5d"]

    changed = False
    if hit < 60 and dd < -4:
        cfg["oversold_gate"]["rsi14_max"] = max(26, cfg["oversold_gate"]["rsi14_max"] - 2)
        cfg["oversold_gate"]["stoch14_max"] = max(15, cfg["oversold_gate"]["stoch14_max"] - 2)
        changed = True
        note = f"Ajuste automático: apertar gates (hit {hit:.1f}%, dd {dd:.1f}%)."
    elif hit > 75:
        cfg["oversold_gate"]["rsi14_max"] = min(36, cfg["oversold_gate"]["rsi14_max"] + 1)
        changed = True
        note = f"Ajuste automático: relaxar ligeiro RSI (hit {hit:.1f}%)."

    if changed:
        cfg["model_version"] = cfg["model_version"] + "_tuned"
    return cfg, note

def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    df = load_history()
    metrics = evaluate_week(df)
    cfg = load_config()
    cfg2, note = propose_small_adjustments(cfg, metrics)

    if cfg2 != cfg:
        save_config(cfg2)

    text = ["<b>WEEKLY REVIEW — Oversold Scanner</b>"]
    text.append(f"Semana (últimos 7 dias):")
    if not metrics.get("ok"):
        text.append(f"— {metrics.get('reason')}")
    else:
        text.append(f"N={metrics['n']} | Hit-rate: {metrics['hit_rate']:.1f}%")
        text.append(f"Rebound médio 3d: {metrics['avg_rebound_3d']:.2f}%")
        text.append(f"Drawdown médio 5d: {metrics['avg_dd_5d']:.2f}%")
        text.append(f"Melhor rebound 3d: {metrics['best_rebound']:.2f}%")
        text.append(f"Pior DD 5d: {metrics['worst_dd']:.2f}%")
    text.append("")
    text.append(note)
    send_telegram_message(token, chat_id, "\n".join(text))

if __name__ == "__main__":
    main()
