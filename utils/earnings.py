from datetime import datetime, timedelta, timezone
import yfinance as yf

def next_earnings_date(symbol: str):
    """
    Returns datetime (UTC) if available, else None.
    yfinance 'calendar' is inconsistent; we treat missing as None.
    """
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal is None or cal.empty:
            return None

        # Common keys: 'Earnings Date'
        if "Earnings Date" not in cal.index:
            return None
        val = cal.loc["Earnings Date"].values
        if val is None or len(val) == 0:
            return None

        # Sometimes it's array of datetimes
        dt = val[0]
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        return None
    except Exception:
        return None

def is_earnings_soon(symbol: str, within_days: int) -> (bool, str):
    dt = next_earnings_date(symbol)
    if dt is None:
        return False, "missing"
    now = datetime.now(timezone.utc)
    if dt <= now + timedelta(days=within_days):
        return True, dt.date().isoformat()
    return False, dt.date().isoformat()
