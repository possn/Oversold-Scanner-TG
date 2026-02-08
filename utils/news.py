import yfinance as yf

def latest_news(symbol: str):
    """
    Returns (headline, publisher, link) or (None,None,None)
    Uses yfinance news if available. If not, return Nones.
    """
    try:
        t = yf.Ticker(symbol)
        items = getattr(t, "news", None)
        if not items:
            return None, None, None
        it = items[0]
        title = it.get("title")
        pub = it.get("publisher")
        link = it.get("link")
        return title, pub, link
    except Exception:
        return None, None, None
