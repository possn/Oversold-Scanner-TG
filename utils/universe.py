import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

WIKI = "https://en.wikipedia.org/wiki/"

def _get_table(url: str) -> pd.DataFrame:
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "lxml")
    tables = pd.read_html(str(soup))
    # heuristic: largest table
    tables.sort(key=lambda t: t.shape[0], reverse=True)
    return tables[0]

def _map_euronext_suffix(exchange: str) -> str:
    # Yahoo suffix mapping (common)
    ex = (exchange or "").strip().upper()
    # Amsterdam
    if ex in ["AMS", "AEX", "EAM"]:
        return ".AS"
    # Paris
    if ex in ["EPA", "PAR", "PA"]:
        return ".PA"
    # Brussels
    if ex in ["EBR", "BRU", "BR"]:
        return ".BR"
    # Lisbon
    if ex in ["ELI", "LIS", "LS"]:
        return ".LS"
    return ""

def universe_sp500():
    df = _get_table(WIKI + "List_of_S%26P_500_companies")
    # columns: Symbol, Security...
    symbols = df["Symbol"].astype(str).tolist()
    return sorted(set(symbols)), "Wikipedia: List_of_S%26P_500_companies"

def universe_nasdaq100():
    df = _get_table(WIKI + "NASDAQ-100")
    # table contains "Ticker" often
    col = None
    for c in df.columns:
        if str(c).lower() in ["ticker", "ticker symbol", "symbol"]:
            col = c
            break
    if col is None:
        # fallback: first column
        col = df.columns[0]
    symbols = df[col].astype(str).tolist()
    symbols = [s.replace(".", "-") for s in symbols]  # yahoo often uses -
    return sorted(set(symbols)), "Wikipedia: NASDAQ-100"

def universe_eurostoxx50():
    df = _get_table(WIKI + "EURO_STOXX_50")
    # usually has columns: "Company", "Ticker", "Industry", "Country"
    col = None
    for c in df.columns:
        if "ticker" in str(c).lower():
            col = c
            break
    if col is None:
        col = df.columns[0]
    syms = df[col].astype(str).tolist()
    # Many tickers not directly Yahoo; keep as-is; will fail if not Yahoo-compatible
    return sorted(set(syms)), "Wikipedia: EURO_STOXX_50"

def universe_euronext100():
    df = _get_table(WIKI + "Euronext_100")
    # often has columns: "Ticker", "Exchange"
    ticker_col = None
    ex_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "ticker" in lc or "symbol" in lc:
            ticker_col = c
        if "exchange" in lc:
            ex_col = c
    if ticker_col is None:
        ticker_col = df.columns[0]

    out = []
    for _, r in df.iterrows():
        t = str(r[ticker_col]).strip()
        ex = str(r[ex_col]).strip() if ex_col else ""
        suf = _map_euronext_suffix(ex)
        if suf and not t.endswith(suf):
            out.append(t + suf)
        else:
            out.append(t)
    return sorted(set(out)), "Wikipedia: Euronext_100"

def universe_psi20():
    df = _get_table(WIKI + "PSI-20")
    # try to find ticker column
    ticker_col = None
    for c in df.columns:
        if "ticker" in str(c).lower() or "symbol" in str(c).lower():
            ticker_col = c
            break
    if ticker_col is None:
        ticker_col = df.columns[0]
    syms = []
    for x in df[ticker_col].astype(str).tolist():
        t = x.strip()
        # ensure Lisbon suffix
        if not t.endswith(".LS"):
            t = t + ".LS"
        syms.append(t)
    return sorted(set(syms)), "Wikipedia: PSI-20"

def build_universe(keys):
    symbols = []
    sources = []
    for k in keys:
        if k == "sp500":
            s, src = universe_sp500()
        elif k == "nasdaq100":
            s, src = universe_nasdaq100()
        elif k == "eurostoxx50":
            s, src = universe_eurostoxx50()
        elif k == "euronext100":
            s, src = universe_euronext100()
        elif k == "psi20":
            s, src = universe_psi20()
        else:
            continue
        symbols.extend(s)
        sources.append(src)

    # dedupe, basic cleanup
    symbols = [re.sub(r"\s+", "", x) for x in symbols]
    symbols = [x for x in symbols if x and x.lower() != "nan"]
    symbols = sorted(set(symbols))
    return symbols, sources
