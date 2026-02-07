import pandas as pd

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"
WIKI_PSI20 = "https://en.wikipedia.org/wiki/PSI-20"
WIKI_ENX100 = "https://en.wikipedia.org/wiki/Euronext_100"

def get_sp500_symbols() -> list[str]:
    tables = pd.read_html(WIKI_SP500)
    df = tables[0]
    syms = df["Symbol"].astype(str).tolist()
    return [s.replace(".", "-").strip() for s in syms]  # Yahoo format (BRK.B -> BRK-B)

def get_nasdaq100_symbols() -> list[str]:
    tables = pd.read_html(WIKI_NDX)
    # a tabela varia; a mais comum tem coluna "Ticker"
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if "ticker" in cols or "ticker symbol" in cols:
            col = t.columns[cols.index("ticker")] if "ticker" in cols else t.columns[cols.index("ticker symbol")]
            syms = t[col].astype(str).tolist()
            return [s.replace(".", "-").strip() for s in syms]
    raise RuntimeError("Não consegui obter Nasdaq-100 da Wikipedia (estrutura mudou).")

def get_psi20_symbols() -> list[str]:
    tables = pd.read_html(WIKI_PSI20)
    # normalmente há coluna "Ticker" ou similar; muitos têm sufixo .LS no Yahoo
    # Vamos inferir: se não tiver sufixo, acrescenta .LS
    best = None
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if "ticker" in cols or "code" in cols or "symbol" in cols:
            best = t
            break
    if best is None:
        raise RuntimeError("Não consegui obter PSI-20 da Wikipedia.")
    # tentar coluna
    for cand in ["Ticker", "Code", "Symbol"]:
        if cand in best.columns:
            raw = best[cand].astype(str).tolist()
            syms = []
            for s in raw:
                s = s.strip()
                if s.endswith(".LS") or s.endswith(".PA") or s.endswith(".AS"):
                    syms.append(s)
                else:
                    # PSI20 costuma ser Lisboa
                    syms.append(s + ".LS")
            return syms
    # fallback: primeira coluna
    raw = best.iloc[:, 0].astype(str).tolist()
    return [s.strip() + ".LS" for s in raw]

def get_euronext100_symbols() -> list[str]:
    tables = pd.read_html(WIKI_ENX100)
    # tabela típica tem coluna "Ticker" com sufixos .PA .AS etc
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if "ticker" in cols:
            raw = t[t.columns[cols.index("ticker")]].astype(str).tolist()
            return [s.strip() for s in raw if s.strip() and s.strip() != "nan"]
    raise RuntimeError("Não consegui obter Euronext 100 da Wikipedia.")
