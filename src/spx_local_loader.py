import yfinance as yf
import pandas as pd
import numpy as np
import py7zr, os, sys, math
from datetime import datetime
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(__file__))
from vix_pipeline_local import (
    compute_vix_variance,
    build_30day_skew,
    compute_forward,
    run_decomposition,
    get_vol_at_strike,
    _signed_delta,
    _bucket_weighted_avg_vol_change,
    get_strikes_in_delta_bucket,
    bucket_raw_contribution,
)


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


def load_spx_options(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "spx_eod")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        df.columns = [c.strip().strip("[]") for c in df.columns]
        for col in ["QUOTE_DATE", "EXPIRE_DATE"]:
            df[col] = df[col].str.strip()
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data = data[data["DTE"] > 0].reset_index(drop=True)
    data["QUOTE_DATE"] = pd.to_datetime(data["QUOTE_DATE"])
    data["EXPIRE_DATE"] = pd.to_datetime(data["EXPIRE_DATE"])
    data = data.sort_values(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]).reset_index(drop=True)
    return data


def find_nearest_expiries(chain, quote_date, target_dte=30):
    day = chain[chain["QUOTE_DATE"] == quote_date]
    exps = (
        day.groupby("EXPIRE_DATE")
        .agg(DTE=("DTE", "first"))
        .reset_index()
        .sort_values("DTE")
    )
    near_rows = exps[exps["DTE"] <= target_dte]
    far_rows = exps[exps["DTE"] > target_dte]
    if near_rows.empty or far_rows.empty:
        return None
    near_row = near_rows.iloc[-1]
    far_row = far_rows.iloc[0]
    return near_row["EXPIRE_DATE"], int(near_row["DTE"]), far_row["EXPIRE_DATE"], int(far_row["DTE"])


def build_chain_df(chain, quote_date, expire_date):
    rows = chain[
        (chain["QUOTE_DATE"] == quote_date) & (chain["EXPIRE_DATE"] == expire_date)
    ].copy()
    rows = rows.sort_values("STRIKE").reset_index(drop=True)
    # Convert bid/ask to numeric before arithmetic — raw data may have strings
    rows["C_BID"] = pd.to_numeric(rows["C_BID"], errors="coerce")
    rows["C_ASK"] = pd.to_numeric(rows["C_ASK"], errors="coerce")
    rows["P_BID"] = pd.to_numeric(rows["P_BID"], errors="coerce")
    rows["P_ASK"] = pd.to_numeric(rows["P_ASK"], errors="coerce")
    rows["cmid"] = (rows["C_BID"] + rows["C_ASK"]) / 2.0
    rows["pmid"] = (rows["P_BID"] + rows["P_ASK"]) / 2.0
    rows = rows.rename(columns={
        "STRIKE": "strike",
        "C_BID": "cbid", "C_ASK": "cask",
        "P_BID": "pbid", "P_ASK": "pask",
        "C_IV": "civ_raw", "P_IV": "piv_raw",
    })
    rows["civ_raw"] = pd.to_numeric(rows["civ_raw"], errors="coerce")
    rows["piv_raw"] = pd.to_numeric(rows["piv_raw"], errors="coerce")
    # Store raw bid/ask for zero-bid truncation check
    rows["cbid_raw"] = rows["cbid"].values.copy()
    rows["cask_raw"] = rows["cask"].values.copy()
    rows["pbid_raw"] = rows["pbid"].values.copy()
    rows["pask_raw"] = rows["pask"].values.copy()
    # Preserve delta columns if present
    for col in ["C_DELTA", "P_DELTA"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows


def compute_forward_local(df_chain, spot, rfr, T):
    if df_chain.empty or "cmid" not in df_chain.columns:
        return spot
    pc_diff = (df_chain["cmid"] - df_chain["pmid"]).abs()
    if pc_diff.empty or pc_diff.isnull().all():
        return spot
    idx = pc_diff.idxmin()
    K_atmf = float(df_chain.loc[idx, "strike"])
    cmid = float(df_chain.loc[idx, "cmid"])
    pmid = float(df_chain.loc[idx, "pmid"])
    if math.isnan(cmid):
        cmid = 0.0
    if math.isnan(pmid):
        pmid = 0.0
    return K_atmf + math.exp(rfr * T) * (cmid - pmid)


def compute_vix_for_date(chain, quote_date, rfr_lookup):
    rfr = rfr_lookup.get(quote_date, 0.045)
    result = find_nearest_expiries(chain, quote_date)
    if result is None:
        return None
    near_exp, dte_near, far_exp, dte_far = result
    spot = float(chain[chain["QUOTE_DATE"] == quote_date]["UNDERLYING_LAST"].iloc[0])
    df_near = build_chain_df(chain, quote_date, near_exp)
    df_far = build_chain_df(chain, quote_date, far_exp)
    T_near = dte_near / 365.0
    T_far = dte_far / 365.0
    F_near = compute_forward_local(df_near, spot, rfr, T_near)
    F_far = compute_forward_local(df_far, spot, rfr, T_far)
    var_near, _, _ = compute_vix_variance(df_near, F_near, rfr, T_near)
    var_far, _, _ = compute_vix_variance(df_far, F_far, rfr, T_far)
    var30 = ((dte_far - 30) * var_near + (30 - dte_near) * var_far) / (dte_far - dte_near)
    sigma_30d = math.sqrt(var30) * 100.0
    T30 = 30.0 / 365.0
    put_skew_30d, call_skew_30d = build_30day_skew(
        df_near, df_far, dte_near, dte_far, spot, F_near, F_far, rfr
    )
    return {
        "date": str(quote_date.date()),
        "spot": spot,
        "near_exp": str(near_exp.date()),
        "far_exp": str(far_exp.date()),
        "DTE_near": dte_near,
        "DTE_far": dte_far,
        "F_near": F_near,
        "F_far": F_far,
        "sigma_30d": sigma_30d,
        "rfr": rfr,
        "put_skew_30d": put_skew_30d,
        "call_skew_30d": call_skew_30d,
        "chain1_df": df_near,
        "chain2_df": df_far,
    }


def main():
    print("Loading data...")
    data = load_spx_options()
    dates = sorted(data["QUOTE_DATE"].unique())
    print(f"Loaded {len(data)} rows, {len(dates)} dates: {dates[0].date()} to {dates[-1].date()}")

    print("Fetching RFR...")
    rfr_lookup = fetch_rfr(dates)
    rfr_vals = [v for v in rfr_lookup.values() if v > 0]
    print(f"RFR: {min(rfr_vals):.4f} to {max(rfr_vals):.4f}")

    print("Computing VIX for all dates...")
    results = []
    for i, dt in enumerate(dates):
        res = compute_vix_for_date(data, dt, rfr_lookup)
        if res:
            results.append(res)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(dates)} dates done")

    print("Fetching actual VIX from yfinance...")
    vix_lookup = fetch_vix_actual(dates)
    vix_vals = [v for v in vix_lookup.values() if v and v > 0]
    if vix_vals:
        print(f"VIX actual: {min(vix_vals):.2f} to {max(vix_vals):.2f}")
    # Attach VIX_actual to each result
    for res in results:
        dt = pd.to_datetime(res["date"])
        res["VIX_actual"] = vix_lookup.get(dt)

    print(f"VIX computed for {len(results)} dates")
    print("\nFirst 5 VIX values:")
    for r in results[:5]:
        print(f"  {r['date']}: SPX={r['spot']:.2f}, VIX={r['sigma_30d']:.2f}, "
              f"near={r['near_exp']}({r['DTE_near']}d), far={r['far_exp']}({r['DTE_far']}d), "
              f"RFR={r['rfr']:.4f}")

    print("\nDecomposition for consecutive pairs...")
    decomp_rows = []
    for i in range(1, len(results)):
        prev, curr = results[i - 1], results[i]
        decomp = run_decomposition(prev, curr)
        if decomp:
            row = {
                "date": curr["date"],
                "SPX_spot": curr["spot"],
                "VIX_computed": curr["sigma_30d"],
                "far_exp": curr["far_exp"],
                "DTE_near": curr["DTE_near"],
                "DTE_far": curr["DTE_far"],
                "F_near": curr["F_near"],
                "F_far": curr["F_far"],
                "sigma_30d": curr["sigma_30d"],
                "rfr": curr["rfr"],
                "F1_sticky_strike": decomp.factor1_sticky_strike,
                "F2_parallel_shift": decomp.factor2_parallel_shift,
                "F3_put_skew_grad": decomp.factor3_put_skew_grad,
                "F4_call_skew_grad": decomp.factor4_call_skew_grad,
                "F5_downside_conv": decomp.factor5_downside_conv,
                "F6_upside_conv": decomp.factor6_upside_conv,
                "sum_factors": sum([
                    decomp.factor1_sticky_strike,
                    decomp.factor2_parallel_shift,
                    decomp.factor3_put_skew_grad,
                    decomp.factor4_call_skew_grad,
                    decomp.factor5_downside_conv,
                    decomp.factor6_upside_conv,
                ]),
                "delta_vix": curr["sigma_30d"] - prev["sigma_30d"],
                "VIX_actual": curr.get("VIX_actual"),
                "delta_VIX_actual": (curr.get("VIX_actual") or 0) - (prev.get("VIX_actual") or 0),
            }
            decomp_rows.append(row)

    df = pd.DataFrame(decomp_rows)
    print(f"\nDecomposition: {len(df)} rows")
    print("\nFirst 5 rows (F1-F6, sum, delta_vix):")
    print(df[["date","VIX_computed","F1_sticky_strike","F2_parallel_shift",
               "F3_put_skew_grad","F4_call_skew_grad","F5_downside_conv",
               "F6_upside_conv","sum_factors","delta_vix"]].head(5).to_string(index=False))

    residual = df["sum_factors"] - df["delta_vix"]

    print(f"\n{'='*60}")
    print(f"OVERALL GOAL: sum(F1..F6) should equal delta_VIX")
    print(f"{'='*60}")
    print(f"Residual stats (sum_factors - delta_vix):")
    print(f"  Mean:   {residual.mean():.4f}")
    print(f"  Std:    {residual.std():.4f}")
    print(f"  Target: mean |residual| < 0.15")
    print(f"")
    print(f"First 5 factor decompositions:")
    print(f"  {'Date':<12} │ {'VIX':>7} │ {'delta_VIX':>10} │ {'sum_F1-F6':>10} │ {'Residual':>10}")
    print(f"  {'-'*10}─┼────{'─'*7}─┼────{'─'*10}─┼────{'─'*10}─┼────{'─'*10}")
    for _, row in df.head(5).iterrows():
        res_val = row["sum_factors"] - row["delta_vix"]
        print(f"  {row['date']:<12} │ {row['VIX_computed']:>7.4f} │ {row['delta_vix']:>+10.4f} │ {row['sum_factors']:>+10.4f} │ {res_val:>+10.4f}")
    print(f"{'='*60}")

    # Reorder columns: date, SPX_spot, VIX_computed, VIX_actual, sum_factors, delta_vix, delta_VIX_actual, then rest
    priority = ["date", "SPX_spot", "VIX_computed", "VIX_actual", "sum_factors", "delta_vix", "delta_VIX_actual"]
    rest = [c for c in df.columns if c not in priority]
    df = df[priority + rest]

    out_path = os.path.join(os.path.dirname(__file__), "output", "vix_decomposition_local_N(d1).csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
