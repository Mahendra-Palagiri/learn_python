'''
  1) Efficiently flatten a multi-dimensional NumPy array
  2) Pandas pivot table with multiple aggregations
  3) Detect and handle outliers in a DataFrame column
  4) Custom Seaborn plot combining multiple plot types (optional, demo guarded)
  5) Optimized vectorized data pipeline (NumPy + Pandas)
'''

from __future__ import annotations
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1) Efficiently flatten a multi-dimensional NumPy array
# ------------------------------------------------------------
def flatten_fast(arr: np.ndarray) -> np.ndarray:
    """
    Return a 1D representation of arr.
    Uses np.ravel which returns a view (no copy) when possible;
    otherwise it falls back to a copy. Equivalent to arr.reshape(-1) for C-contiguous arrays.
    """
    return np.ravel(arr)


# ------------------------------------------------------------
# 2) Pandas pivot table with multiple aggregations
# ------------------------------------------------------------
def make_multiagg_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a pivot table summarizing Sales and Cost by Dept x Quarter with multiple agg funcs.
    """
    pivot = pd.pivot_table(
        df,
        index="Dept",
        columns="Quarter",
        values=["Sales", "Cost"],
        aggfunc={"Sales": ["sum", "mean", "count"], "Cost": ["sum", "mean"]},
        fill_value=0,
        margins=True,
        margins_name="Total",
    )
    return pivot


# ------------------------------------------------------------
# 3) Detect and handle outliers using IQR (winsorization by capping)
# ------------------------------------------------------------
def cap_outliers_iqr(s: pd.Series, k: float = 1.5) -> tuple[pd.Series, float, float]:
    """
    Detect outliers via IQR rule and cap values to [low, high].
    Returns (capped_series, low, high).
    """
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return s.clip(lower=low, upper=high), low, high


# ------------------------------------------------------------
# 4) Custom Seaborn plot combining scatter + regression + KDE contours
#    (Optional: demo guarded to avoid hard dependency on seaborn/matplotlib)
# ------------------------------------------------------------
def custom_combo_plot(df: pd.DataFrame, x: str = "x", y: str = "y", hue: str | None = "group"):
    """
    Return a Matplotlib Axes with layered Seaborn plot (scatter + reg + kde).
    If seaborn/matplotlib are unavailable, returns None.
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # 1) scatter (colored by group if provided)
    if hue is not None and hue in df.columns:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.35, edgecolor=None, ax=ax)
    else:
        sns.scatterplot(data=df, x=x, y=y, alpha=0.35, edgecolor=None, ax=ax)

    # 2) regression line (no scatter)
    sns.regplot(data=df, x=x, y=y, scatter=False, ci=None, ax=ax, line_kws={"lw": 2})

    # 3) 2D density contours
    sns.kdeplot(data=df, x=x, y=y, levels=6, linewidths=1, ax=ax)

    ax.set_title("Custom Combo Plot: Scatter + Regression + KDE Contours")
    ax.grid(True, alpha=0.2)
    return ax


# ------------------------------------------------------------
# 5) Optimized vectorized data pipeline (NumPy + Pandas)
# ------------------------------------------------------------
def vectorized_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given order-line data with columns:
        order_id, date, sku, qty, price, discount, tax_rate
    Compute gross, net, total, revenue bands, month aggregation, and a SKU pivot.
    Returns (monthly_agg, sku_pivot).
    """
    # 1) Dates
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 2) Vectorized math
    gross = df["qty"] * df["price"]
    net = gross * (1.0 - df["discount"])
    total = net * (1.0 + df["tax_rate"])
    df["gross"] = gross
    df["net"] = net
    df["total"] = total

    # 3) Bands
    conds = [
        df["total"] < 50,
        (df["total"] >= 50) & (df["total"] < 150),
        df["total"] >= 150,
    ]
    labels = ["low", "medium", "high"]
    df["band"] = np.select(conds, labels, default="low")

    # 4) Month (Period for tidy grouping)
    df["month"] = df["date"].dt.to_period("M")

    # 5) Aggregations
    monthly = (
        df.groupby("month", as_index=False)
          .agg(
              total_revenue=("total", "sum"),
              orders=("order_id", "nunique"),
              avg_line_total=("total", "mean"),
          )
          .sort_values("month")
    )

    sku_pivot = pd.pivot_table(
        df,
        index="month",
        columns="sku",
        values="total",
        aggfunc="sum",
        fill_value=0,
    )

    return monthly, sku_pivot


# ------------------------------------------------------------
# Demo / quick test when running as a script
# ------------------------------------------------------------
def _demo():
    print("1) flatten_fast demo")
    a = np.arange(12).reshape(3, 4)
    print("input shape:", a.shape, "-> output:", flatten_fast(a), "; ndim:", flatten_fast(a).ndim)

    print("\n2) pivot table demo")
    df_pivot_src = pd.DataFrame({
        "Dept":    ["IT","IT","IT","HR","HR","Sales","Sales","Sales"],
        "Quarter": ["Q1","Q1","Q2","Q1","Q2","Q1","Q2","Q2"],
        "Sales":   [100,120,130,80,90,200,220,180],
        "Cost":    [60,  70,  75, 50,55,120,130,110]
    })
    pv = make_multiagg_pivot(df_pivot_src)
    print(pv)

    print("\n3) outlier capping demo")
    rng = np.random.default_rng(42)
    s = pd.Series(np.concatenate([rng.normal(50, 10, 1000), [300, -50]]))
    capped, low, high = cap_outliers_iqr(s)
    print("thresholds:", round(low,2), round(high,2))
    print("original std ~", round(s.std(),2), " -> capped std ~", round(capped.std(),2))

    print("\n4) custom plot demo (skipped if seaborn/matplotlib missing)")
    try:
        import seaborn as sns  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        x = rng.normal(0, 1, 600)
        y = 2.0*x + rng.normal(0, 1.0, 600)
        cat = rng.choice(["A","B"], size=600, p=[0.6, 0.4])
        df_plot = pd.DataFrame({"x": x, "y": y, "group": cat})
        ax = custom_combo_plot(df_plot)
        if ax is not None:
            plt.tight_layout()
            # Comment the next line if running headless
            # plt.show()
            print("Custom plot created (not displayed in headless mode).")
        else:
            print("Seaborn/matplotlib not available; plot skipped.")
    except Exception as e:
        print("Plot skipped due to error:", e)

    print("\n5) vectorized pipeline demo")
    df_pipe = pd.DataFrame({
        "order_id":  [1,1,2,2,3,3,4],
        "date":      ["2025-01-02","2025-01-02","2025-02-10","2025-02-10","2025-02-15","2025-02-15","2025-03-01"],
        "sku":       ["A","B","A","C","B","C","A"],
        "qty":       [2, 1, 3, 1, 5, 2, 4],
        "price":     [10.0, 25.0, 10.0, 40.0, 25.0, 40.0, 10.0],
        "discount":  [0.10, 0.00, 0.15, 0.10, 0.00, 0.05, 0.20],
        "tax_rate":  [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
    })
    monthly, sku_pivot = vectorized_pipeline(df_pipe)
    print("Monthly summary:\n", monthly)
    print("\nRevenue by SKU (pivot):\n", sku_pivot)


if __name__ == "__main__":
    _demo()
