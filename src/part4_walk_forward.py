# =============================================================================
# part4_walk_forward.py — Walk-Forward Backtesting
# Swiss Multi-Asset Portfolio Optimizer
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    fetch_smi, compute_returns, covariance_matrix,
    annualized_return, portfolio_performance,
    RISK_FREE_RATE, TRADING_DAYS, SMI_TICKERS
)

FIGURES_PATH = "reports/figures/"
os.makedirs(FIGURES_PATH, exist_ok=True)


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

def optimize_max_sharpe(returns: pd.DataFrame) -> np.ndarray:
    n           = returns.shape[1]
    cov         = covariance_matrix(returns).values
    mu          = annualized_return(returns).values
    w0          = np.ones(n) / n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * n

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        if vol < 1e-8:
            return 0.0
        return -(ret - RISK_FREE_RATE) / vol

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 1000})
    return result.x


def walk_forward_backtest(prices: pd.DataFrame,
                           train_years: int = 2,
                           test_months: int = 6) -> pd.DataFrame:
    """
    Rolling walk-forward backtest.

    Parameters
    ----------
    prices       : DataFrame of adjusted close prices
    train_years  : calibration window in years
    test_months  : out-of-sample test window in months

    Returns
    -------
    DataFrame with columns: date, strategy_return, equal_weight_return,
                             window_id, weights_*
    """
    returns  = compute_returns(prices)
    tickers  = list(returns.columns)
    n        = len(tickers)

    train_days = train_years * TRADING_DAYS
    test_days  = test_months * 21  # ~21 trading days/month

    results  = []
    window   = 0

    i = train_days
    while i + test_days <= len(returns):
        # --- Training window ---
        train_ret = returns.iloc[i - train_days: i]

        # --- Optimize on training data ---
        try:
            w_opt = optimize_max_sharpe(train_ret)
        except Exception:
            w_opt = np.ones(n) / n

        w_eq = np.ones(n) / n

        # --- Test window ---
        test_ret  = returns.iloc[i: i + test_days]
        test_dates = test_ret.index

        for date, row in zip(test_dates, test_ret.values):
            opt_ret = float(row @ w_opt)
            eq_ret  = float(row @ w_eq)
            record  = {
                "date":            date,
                "opt_return":      opt_ret,
                "eq_return":       eq_ret,
                "window_id":       window,
                "train_start":     returns.index[i - train_days],
                "train_end":       returns.index[i - 1],
            }
            for j, t in enumerate(tickers):
                record[f"w_{t}"] = w_opt[j]
            results.append(record)

        i += test_days
        window += 1

    return pd.DataFrame(results).set_index("date")


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def compute_metrics(returns_series: pd.Series) -> dict:
    """Annualized metrics from a daily return series."""
    ann_ret  = returns_series.mean() * TRADING_DAYS
    ann_vol  = returns_series.std()  * np.sqrt(TRADING_DAYS)
    sharpe   = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum      = (1 + returns_series).cumprod()
    peak     = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd   = drawdown.min()

    # Hit rate
    hit_rate = (returns_series > 0).mean()

    return {
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "hit_rate":     hit_rate,
    }


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_walk_forward_results(wf: pd.DataFrame, prices: pd.DataFrame):
    """Main walk-forward results dashboard."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # Cumulative returns
    cum_opt = (1 + wf["opt_return"]).cumprod()
    cum_eq  = (1 + wf["eq_return"]).cumprod()

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0d1117")
    ax1.plot(cum_opt.index, cum_opt.values,
             color="#f1c40f", linewidth=2, label="Walk-Forward Max-Sharpe (OOS)")
    ax1.plot(cum_eq.index,  cum_eq.values,
             color="#e74c3c", linewidth=1.5, linestyle="--", label="Equal-Weight (OOS)")

    # Shade windows
    windows = wf["window_id"].unique()
    colors  = ["#1a2a1a", "#0d1a2a"]
    for wid in windows:
        mask = wf["window_id"] == wid
        dates = wf.index[mask]
        ax1.axvspan(dates[0], dates[-1],
                    alpha=0.15, color=colors[wid % 2], zorder=0)

    ax1.set_title("Walk-Forward Backtest — Out-of-Sample Cumulative Returns",
                  color="white", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#1e2a3a", edgecolor="#444",
               labelcolor="white", fontsize=9)
    ax1.axhline(1, color="#555", linestyle=":", linewidth=1)
    for spine in ax1.spines.values(): spine.set_edgecolor("#444")

    # Rolling Sharpe (252-day)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0d1117")
    roll_sharpe_opt = (wf["opt_return"].rolling(252).mean() * TRADING_DAYS -
                       RISK_FREE_RATE) / (wf["opt_return"].rolling(252).std() *
                                          np.sqrt(TRADING_DAYS))
    roll_sharpe_eq  = (wf["eq_return"].rolling(252).mean() * TRADING_DAYS -
                       RISK_FREE_RATE) / (wf["eq_return"].rolling(252).std() *
                                          np.sqrt(TRADING_DAYS))
    ax2.plot(roll_sharpe_opt.index, roll_sharpe_opt,
             color="#f1c40f", linewidth=1.5, label="Max-Sharpe")
    ax2.plot(roll_sharpe_eq.index,  roll_sharpe_eq,
             color="#e74c3c", linewidth=1.2, linestyle="--", label="Equal-Weight")
    ax2.axhline(0, color="#555", linewidth=0.8)
    ax2.set_title("Rolling 1-Year Sharpe Ratio", color="white", fontsize=10)
    ax2.set_ylabel("Sharpe Ratio", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1e2a3a", edgecolor="#444",
               labelcolor="white", fontsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")

    # Drawdown
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0d1117")
    dd_opt = (cum_opt / cum_opt.cummax() - 1) * 100
    dd_eq  = (cum_eq  / cum_eq.cummax()  - 1) * 100
    ax3.fill_between(dd_opt.index, dd_opt, 0,
                     color="#f1c40f", alpha=0.4, label="Max-Sharpe")
    ax3.fill_between(dd_eq.index,  dd_eq,  0,
                     color="#e74c3c", alpha=0.3, label="Equal-Weight")
    ax3.set_title("Drawdown (%)", color="white", fontsize=10)
    ax3.set_ylabel("Drawdown (%)", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#1e2a3a", edgecolor="#444",
               labelcolor="white", fontsize=8)
    for spine in ax3.spines.values(): spine.set_edgecolor("#444")

    # Weight evolution (stacked area)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor("#0d1117")
    tickers = [c.replace("w_", "") for c in wf.columns if c.startswith("w_")]
    labels  = [SMI_TICKERS.get(t, t) for t in tickers]
    colors_w = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    weight_data = [wf[f"w_{t}"].values for t in tickers]
    ax4.stackplot(wf.index, weight_data,
                  labels=labels, colors=colors_w, alpha=0.75)
    ax4.set_title("Portfolio Weight Evolution Across Windows",
                  color="white", fontsize=10)
    ax4.set_ylabel("Weight", color="white")
    ax4.tick_params(colors="white")
    ax4.set_ylim(0, 1)
    ax4.legend(facecolor="#1e2a3a", edgecolor="#444",
               labelcolor="white", fontsize=8,
               loc="upper left", ncol=3)
    for spine in ax4.spines.values(): spine.set_edgecolor("#444")

    plt.suptitle("Walk-Forward Backtest — SMI Portfolio (2-Year Train / 6-Month Test)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    path = FIGURES_PATH + "10_walk_forward_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_window_performance(wf: pd.DataFrame):
    """Per-window Sharpe comparison: OOS Max-Sharpe vs Equal-Weight."""
    windows = sorted(wf["window_id"].unique())
    sharpes_opt, sharpes_eq, labels = [], [], []

    for wid in windows:
        mask  = wf["window_id"] == wid
        chunk = wf[mask]
        m_opt = compute_metrics(chunk["opt_return"])
        m_eq  = compute_metrics(chunk["eq_return"])
        sharpes_opt.append(m_opt["sharpe"])
        sharpes_eq.append(m_eq["sharpe"])
        start = chunk["train_end"].iloc[0].strftime("%Y-%m")
        labels.append(f"W{wid+1}\n{start}")

    x   = np.arange(len(windows))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(windows) * 1.5), 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars1 = ax.bar(x - w/2, sharpes_opt, w, label="Max-Sharpe (OOS)",
                   color="#f1c40f", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, sharpes_eq,  w, label="Equal-Weight (OOS)",
                   color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars1, sharpes_opt):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.02 if val >= 0 else -0.12),
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="white")
    for bar, val in zip(bars2, sharpes_eq):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.02 if val >= 0 else -0.12),
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="white")

    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=8)
    ax.set_ylabel("Out-of-Sample Sharpe Ratio", color="white")
    ax.set_title("Per-Window OOS Sharpe — Max-Sharpe vs Equal-Weight",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2a3a", edgecolor="#444",
              labelcolor="white", fontsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

    plt.tight_layout()
    path = FIGURES_PATH + "11_window_sharpe_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# RESULTS TABLE
# =============================================================================

def print_results(wf: pd.DataFrame):
    m_opt = compute_metrics(wf["opt_return"])
    m_eq  = compute_metrics(wf["eq_return"])

    print("\n" + "="*65)
    print("  WALK-FORWARD BACKTEST — OUT-OF-SAMPLE RESULTS")
    print("="*65)

    rows = []
    for label, m in [("Max-Sharpe (OOS)", m_opt), ("Equal-Weight (OOS)", m_eq)]:
        rows.append({
            "Strategy":     label,
            "Ann. Return":  f'{m["ann_return"]:.2%}',
            "Ann. Vol":     f'{m["ann_vol"]:.2%}',
            "Sharpe":       f'{m["sharpe"]:.2f}',
            "Max Drawdown": f'{m["max_drawdown"]:.2%}',
            "Hit Rate":     f'{m["hit_rate"]:.2%}',
        })
    print(pd.DataFrame(rows).to_string(index=False))

    n_windows = wf["window_id"].nunique()
    print(f"\nTotal OOS windows : {n_windows}")
    print(f"OOS period        : {wf.index[0].date()} → {wf.index[-1].date()}")
    print("="*65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Fetching SMI data...")
    prices = fetch_smi()

    print("Running walk-forward backtest (2-year train / 6-month test)...")
    wf = walk_forward_backtest(prices, train_years=2, test_months=6)

    print_results(wf)

    print("\nGenerating charts...")
    plot_walk_forward_results(wf, prices)
    plot_window_performance(wf)

    print("\nPart 4 complete. Charts saved in reports/figures/")