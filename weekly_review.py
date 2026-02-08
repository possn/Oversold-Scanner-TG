import json
import os
from datetime import datetime, timezone, timedelta
import pandas as pd

from utils.telegram import send_message

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")

HIST = "data/history.csv"

def load_cfg():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def save_cfg(cfg):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def weekly_window():
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=7)
    return start

def evaluate_week(df_week: pd.DataFrame):
    # Basic success definition:
    # success if max_3d >= entry_close OR max_5d >= entry_close*(1+0.5%) (proxy rebound)
    # but we only compute if those fields already filled (they will fill over time)
    # If missing: "insufficient outcome"
    rows = []
    for _, r in df_week.iterrows():
        entry = r["close"]
        try:
            max3 = float(r["max_3d"]) if str(r["max_3d"]).strip() != "" else None
            max5 = float(r["max_5d"]) if str(r["max_5d"]).strip() != "" else None
            min5 = float(r["min_5d"]) if str(r["min_5d"]).strip() != "" else None
        except Exception:
            max3 = max5 = min5 = None

        if max3 is None or max5 is None or min5 is None:
            rows.append((r["symbol"], r["region"], None, None, None))
            continue

        rebound5 = (max5 / entry) - 1.0
        dd5 = (min5 / entry) - 1.0
        success = rebound5 >= 0.005  # +0.5%
        rows.append((r["symbol"], r["region"], success, rebound5, dd5))
    return rows

def propose_simple_tune(cfg, rows):
    # Only tune if enough labelled outcomes
    labelled = [x for x in rows if x[2] is not None]
    if len(labelled) < cfg["min_sample_for_auto_tune"]:
        return None, f"amostra insuficiente ({len(labelled)}/{cfg['min_sample_for_auto_tune']})"

    success_rate = sum(1 for x in labelled if x[2]) / len(labelled)

    # Minimal, low-complexity adjustments within bounds:
    # - if success low: tighten score_threshold slightly (reduce false positives)
    # - if success high: allow a bit more breadth (lower threshold slightly)
    old_thr = cfg["scoring"]["score_threshold"]
    new_thr = old_thr

    if success_rate < 0.70:
        new_thr = min(7.5, old_thr + 0.25)
    elif success_rate > 0.85:
        new_thr = max(5.0, old_thr - 0.15)

    if new_thr == old_thr:
        return None, f"sem alteração (success_rate={success_rate:.2%})"

    cfg2 = json.loads(json.dumps(cfg))
    cfg2["scoring"]["score_threshold"] = round(new_thr, 2)
    return cfg2, f"score_threshold {old_thr} → {new_thr} (success_rate={success_rate:.2%})"

def main():
    cfg = load_cfg()
    if not os.path.exists(HIST):
        send_message(TOKEN, CHAT, "*Weekly Review* — sem histórico ainda.")
        return

    df = pd.read_csv(HIST)
    start = weekly_window()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    dfw = df[df["ts_utc"] >= start]

    rows = evaluate_week(dfw)

    labelled = [x for x in rows if x[2] is not None]
    if labelled:
        sr = sum(1 for x in labelled if x[2]) / len(labelled)
        avg_reb = sum(x[3] for x in labelled) / len(labelled)
        avg_dd = sum(x[4] for x in labelled) / len(labelled)
        summary = (
            f"*Weekly Review (últimos 7 dias)*\n"
            f"- n={len(labelled)}\n"
            f"- taxa sucesso={sr:.1%}\n"
            f"- rebound médio (5d)={avg_reb:.2%}\n"
            f"- drawdown médio (5d)={avg_dd:.2%}\n"
        )
    else:
        summary = "*Weekly Review (últimos 7 dias)*\n- resultados ainda insuficientes (outcomes em falta)."

    tune_msg = ""
    if cfg.get("auto_tune_enabled", False):
        new_cfg, msg = propose_simple_tune(cfg, rows)
        tune_msg = f"\n*Auto-tune*: {msg}"
        if new_cfg is not None:
            save_cfg(new_cfg)

    # list top failures/successes
    lines = [summary + tune_msg, "\n*Notas*",
             "- Sem invenções: se max/min 5d ainda não preenchidos, não avalia.",
             "- Ajustes só em thresholds simples, sem aumentar complexidade."]

    send_message(TOKEN, CHAT, "\n".join(lines))

if __name__ == "__main__":
    main()
