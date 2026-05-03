import os
import glob
import yfinance as yf
import pandas as pd


def fetch_rfr(dates):
    t = yf.download("^IRX", start="2023-01-01", end="2023-04-01", progress=False)
    close = t["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    rfr_raw = {pd.to_datetime(k): float(v) / 100.0 for k, v in close.to_dict().items()}

    def get_rfr(dt):
        dt = pd.to_datetime(dt)
        if dt in rfr_raw:
            return rfr_raw[dt]
        before = max((d for d in rfr_raw if d <= dt), default=None)
        after = min((d for d in rfr_raw if d >= dt), default=None)
        if before and after and before != after:
            w = (dt - before).days / (after - before).days
            return rfr_raw[before] * (1 - w) + rfr_raw[after] * w
        if before:
            return rfr_raw[before]
        if after:
            return rfr_raw[after]
        raise ValueError(f"No RFR data available for {dt} and no fallback permitted")

    return {dt: get_rfr(dt) for dt in dates}


def fetch_vix_actual(dates):
    """Download actual VIX from yfinance for the given dates."""
    if not dates:
        return {}
    start = min(dates)
    end = max(dates) + pd.Timedelta(days=1)
    t = yf.download("^VIX", start=start, end=end, progress=False)
    close = t["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    vix_raw = {pd.to_datetime(k): float(v) for k, v in close.to_dict().items()}

    def get_vix(dt):
        dt = pd.to_datetime(dt)
        if dt in vix_raw:
            return vix_raw[dt]
        # Interpolate if missing
        before = max((d for d in vix_raw if d <= dt), default=None)
        after = min((d for d in vix_raw if d >= dt), default=None)
        if before and after:
            w = (dt - before).days / max((after - before).days, 1)
            return vix_raw[before] * (1 - w) + vix_raw[after] * w
        return vix_raw.get(before) or vix_raw.get(after)

    return {dt: get_vix(dt) for dt in dates}


def load_spx_options(data_dir=None, quote_dates=None):
    """Load SPX EOD options data from text files.

    `data_dir` may contain spx_eod_*.txt files directly (flat layout)
    or organized into year subdirectories (e.g. data_dir/2023/...).
    The recursive glob picks both up.

    `quote_dates`: optional iterable of date strings ("YYYY-MM-DD") or
    pd.Timestamps. When provided, the returned DataFrame is filtered
    to only those QUOTE_DATE values — useful for event-study runs that
    only need a handful of days. None (default) loads everything.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "spx_eod")
    pattern = os.path.join(data_dir, "**", "spx_eod_*.txt")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No SPX EOD .txt files found under {data_dir}. "
            f"Expected files matching spx_eod_*.txt at any depth."
        )
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().strip("[]") for c in df.columns]
        for col in ["QUOTE_DATE", "EXPIRE_DATE"]:
            df[col] = df[col].str.strip()
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data = data[data["DTE"] > 0].reset_index(drop=True)
    data["QUOTE_DATE"] = pd.to_datetime(data["QUOTE_DATE"])
    data["EXPIRE_DATE"] = pd.to_datetime(data["EXPIRE_DATE"])
    if quote_dates is not None:
        target_dates = {pd.to_datetime(d).normalize() for d in quote_dates}
        data = data[data["QUOTE_DATE"].dt.normalize().isin(target_dates)]
    data = data.sort_values(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]).reset_index(drop=True)
    return data
