import os
import json
import pandas as pd
from datetime import datetime, timezone

from utils.audit import AuditLog
from utils.telegram import send_message
from utils.universe import build_universe
from utils.scanner import (
    scan_symbols, three_resets, pick_top_per_reset,
    intersection_2_of_3, render_report
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")

def load_cfg():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_history():
    os.makedirs("data", exist_ok=True)
    path = "data/history.csv"
    if not os.path.exists(path):
        pd.DataFrame(columns=[
            "ts_utc","region","symbol","score","decision","close",
            "max_3d","max_5d","min_5d"
        ]).to_csv(path, index=False)

def append_history(region: str, reset_tops, inter):
    ensure_history()
    path = "data/history.csv"
    df = pd.read_csv(path)
    ts = datetime.now(timezone.utc).isoformat()

    # store intersection set (best candidates)
    best = {}
    for key in ["A","B","C"]:
        for c in reset_tops[key]:
            if c.symbol not in inter:
                continue
            if (c.symbol not in best) or (c.score > best[c.symbol].score):
                best[c.symbol] = c

    rows = []
    for sym, c in best.items():
        decision = "AGUARDAR_T1" if c.needs_t1 else "EXEC"
        rows.append({
            "ts_utc": ts,
            "region": region,
            "symbol": sym,
            "score": c.score,
            "decision": decision,
            "close": c.close,
            "max_3d": "",
            "max_5d": "",
            "min_5d": ""
        })

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(path, index=False)

def main(region_key: str):
    cfg = load_cfg()
    audit = AuditLog()

    # universe
    universe_keys = cfg["universes"][region_key]
    symbols, sources = build_universe(universe_keys)
    for s in sources:
        audit.add_source(s)
    audit.ok("0_universe_retrieve")

    # three resets
    resets = three_resets(symbols)

    # scan each reset independently (no anchoring)
    reset_tops = {}
    meta_parts = []
    for key in ["A","B","C"]:
        cands, meta = scan_symbols(resets[key], cfg, f"{region_key}/{key}", audit)
        meta_parts.append(meta)
        reset_tops[key] = pick_top_per_reset(cands, cfg["outputs"]["top_per_reset"])

    # intersection
    inter = intersection_2_of_3(reset_tops["A"], reset_tops["B"], reset_tops["C"])
    audit.ok("8_intersection_2of3")

    # t+1 enforced already in candidates
    audit.ok("7_Tplus1")
    audit.ok("9_scoring")

    meta = " | ".join(meta_parts)
    region_name = "EUROPA (EuroStoxx50 + Euronext100 + PSI20)" if region_key == "EU" else "EUA (S&P500 + Nasdaq100)"
    report = render_report(region_name, audit, reset_tops, inter, cfg, meta)

    # persist history
    append_history(region_name, reset_tops, inter)

    # send
    send_message(TOKEN, CHAT, report)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ["EU", "US"]:
        raise SystemExit("Usage: python run_daily.py EU|US")
    main(sys.argv[1])
