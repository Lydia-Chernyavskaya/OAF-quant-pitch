import pandas as pd, numpy as np
from pathlib import Path
CSV = Path(__file__).resolve().parent.parent / "src/output/vix_decomposition_hybrid.csv"
df = pd.read_csv(CSV)
d  = df.dropna(subset=["F1","F2","F3","F4","F5","F6","delta_VIX"]).copy()
d["abs_resid"] = d["residual"].abs()
d["abs_dvix"]  = d["delta_VIX"].abs()

# (1) Where does the worst residual land?
print("=== Top 10 by |residual| ===")
print(d.nlargest(10, "abs_resid")[
    ["date","VIX_computed","delta_VIX","sum_factors","residual","F1","F2","F3","F4","F5","F6"]
].to_string(index=False))

# (2) Residual distribution
print("\n=== |residual| quantiles ===")
print(d["abs_resid"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(3))

# (3) On calm days (|ΔVIX| <= 0.5), how big are the factors and the residual?
calm = d[d["abs_dvix"] <= 0.5]
print(f"\n=== Calm days (|ΔVIX| ≤ 0.5): {len(calm)} of {len(d)} ===")
print("max |F_i| on calm days:")
for c in ["F1","F2","F3","F4","F5","F6"]:
    print(f"  {c}: {calm[c].abs().max():.3f}")
print(f"calm residual: mean={calm['residual'].mean():+.3f}  std={calm['residual'].std():.3f}  max|resid|={calm['abs_resid'].max():.3f}")

# (4) Per-factor std across all days (sanity: not blowing up)
print("\n=== Factor std (all days) ===")
print(d[["F1","F2","F3","F4","F5","F6"]].std().round(3))

# (5) Quick eyeball at our 5 candidate Stage-3 days
candidates = ["2023-03-13","2023-08-02","2023-09-21","2023-10-26","2023-11-14"]
print("\n=== Stage-3 candidate days ===")
print(d[d["date"].isin(candidates)][
    ["date","VIX_actual","delta_VIX","sum_factors","residual","F1","F2","F3","F4","F5","F6"]
].to_string(index=False))