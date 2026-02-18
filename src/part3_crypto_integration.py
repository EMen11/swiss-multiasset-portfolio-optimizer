# =============================================================================
# part3_crypto_integration.py — Crypto as Alternative Asset Class (SMI + BTC/ETH)
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
    fetch_all, fetch_smi, compute_returns, covariance_matrix,
    annualized_return, annualized_volatility, sharpe_ratio,
    portfolio_performance, RISK_FREE_RATE, SMI_TICKERS, CRYPTO_TICKERS
)

FIGURES_PATH = "reports/figures/"
os.makedirs(FIGURES_PATH, exist_ok=True)

ALL_TICKERS = {**SMI_TICKERS, **CRYPTO_TICKERS}


# =============================================================================
# OPTIMIZATION
# =============================================================================

def max_sharpe(returns: pd.DataFrame,
               max_crypto_weight: float = 0.15) -> dict:
    """
    Max-Sharpe optimization with optional crypto weight cap.
    max_crypto_weight : max total allocation to BTC + ETH (default 15%)
    """
    tickers     = list(returns.columns)
    crypto_idx  = [i for i, t in enumerate(tickers) if t in CRYPTO_TICKERS]
    n           = len(tickers)
    cov         = covariance_matrix(returns).values
    mu          = annualized_return(returns).values
    w0          = np.ones(n) / n

    constraints = [
        {"type": "eq",  "fun": lambda w: np.sum(w) - 1},
    ]
    if crypto_idx:
        constraints.append(
            {"type": "ineq",
             "fun": lambda w: max_crypto_weight - np.sum(w[crypto_idx])}
        )

    bounds = [(0, 1)] * n

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        return -(ret - RISK_FREE_RATE) / vol

    result  = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 2000})
    weights = result.x
    perf    = portfolio_performance(weights, returns)
    return {"weights": weights, "performance": perf, "tickers": tickers}


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def plot_correlation_matrix(returns: pd.DataFrame):
    """Heatmap of asset correlations."""
    corr    = returns.corr()
    labels  = [ALL_TICKERS.get(t, t) for t in corr.columns]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color="white", fontsize=9)
    ax.set_yticklabels(labels, color="white", fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if abs(val) > 0.4 else "white")

    ax.set_title("Asset Correlation Matrix — SMI + Crypto (2018–Present)",
                 color="white", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    path = FIGURES_PATH + "07_correlation_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_risk_return_all_assets(returns: pd.DataFrame):
    """Risk-return scatter for all individual assets."""
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    tickers = list(returns.columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for t in tickers:
        is_crypto = t in CRYPTO_TICKERS
        color     = "#f39c12" if is_crypto else "#3498db"
        marker    = "D" if is_crypto else "o"
        size      = 180 if is_crypto else 120
        label_txt = ALL_TICKERS.get(t, t)
        ax.scatter(ann_vol[t], ann_ret[t],
                   color=color, marker=marker, s=size,
                   zorder=5, edgecolors="white", linewidths=0.6)
        ax.annotate(label_txt,
                    xy=(ann_vol[t], ann_ret[t]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=8.5, color="white")

    # Legend proxies
    import matplotlib.patches as mpatches
    eq   = mpatches.Patch(color="#3498db", label="SMI Equities")
    cry  = mpatches.Patch(color="#f39c12", label="Crypto Assets")
    ax.legend(handles=[eq, cry], facecolor="#1e2a3a",
              edgecolor="#444", labelcolor="white", fontsize=9)

    ax.set_xlabel("Annualized Volatility", color="white", fontsize=11)
    ax.set_ylabel("Annualized Return",     color="white", fontsize=11)
    ax.set_title("Risk-Return Profile — SMI + Crypto (2018–Present)",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    path = FIGURES_PATH + "08_risk_return_all_assets.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_portfolio_comparison(smi_only: dict,
                               with_crypto: dict,
                               returns_smi: pd.DataFrame,
                               returns_all: pd.DataFrame):
    """Compare SMI-only vs SMI+Crypto portfolio: weights + cumulative."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- Top left: SMI-only weights ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0d1117")
    tickers_smi = smi_only["tickers"]
    labels_smi  = [SMI_TICKERS.get(t, t) for t in tickers_smi]
    w_smi       = smi_only["weights"] * 100
    bars = ax1.bar(labels_smi, w_smi, color="#f1c40f",
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, w_smi):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=8, color="white")
    p = smi_only["performance"]
    ax1.set_title(f'SMI Only\nR:{p["return"]:.1%} σ:{p["volatility"]:.1%} S:{p["sharpe"]:.2f}',
                  color="white", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Weight (%)", color="white")
    ax1.set_ylim(0, 100)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values(): spine.set_edgecolor("#444")

    # --- Top right: SMI+Crypto weights ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0d1117")
    tickers_all = with_crypto["tickers"]
    labels_all  = [ALL_TICKERS.get(t, t) for t in tickers_all]
    w_all       = with_crypto["weights"] * 100
    colors_all  = ["#f39c12" if t in CRYPTO_TICKERS else "#2ecc71"
                   for t in tickers_all]
    bars2 = ax2.bar(labels_all, w_all, color=colors_all,
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, w_all):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=8, color="white")
    p2 = with_crypto["performance"]
    ax2.set_title(
        f'SMI + Crypto (max 15%)\nR:{p2["return"]:.1%} σ:{p2["volatility"]:.1%} S:{p2["sharpe"]:.2f}',
        color="white", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Weight (%)", color="white")
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors="white", axis="both")
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")

    # --- Bottom: cumulative returns ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#0d1117")

    n_smi = returns_smi.shape[1]
    eq_w  = np.ones(n_smi) / n_smi
    cum_eq = (1 + returns_smi.values @ eq_w).cumprod()
    ax3.plot(returns_smi.index, cum_eq,
             label="Equal-Weight (SMI)", color="#e74c3c",
             linestyle="--", linewidth=1.6)

    cum_smi = (1 + returns_smi.values @ smi_only["weights"]).cumprod()
    ax3.plot(returns_smi.index, cum_smi,
             label="Max-Sharpe (SMI only)", color="#f1c40f",
             linestyle="-.", linewidth=1.6)

    # Align crypto portfolio to common dates
    common_idx = returns_smi.index.intersection(returns_all.index)
    r_all_aligned = returns_all.loc[common_idx]
    cum_crypto = (1 + r_all_aligned.values @ with_crypto["weights"]).cumprod()
    ax3.plot(common_idx, cum_crypto,
             label="Max-Sharpe (SMI + Crypto ≤15%)",
             color="#2ecc71", linestyle="-", linewidth=1.8)

    ax3.set_title("Growth of $1 — SMI Only vs SMI + Crypto",
                  color="white", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Date",                color="white")
    ax3.set_ylabel("Portfolio Value ($)", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#1e2a3a", edgecolor="#444",
               labelcolor="white", fontsize=9)
    ax3.axhline(1, color="#555", linestyle=":", linewidth=1)
    for spine in ax3.spines.values(): spine.set_edgecolor("#444")

    plt.suptitle("Portfolio Optimization — Impact of Crypto Allocation (SMI Universe)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    path = FIGURES_PATH + "09_smi_vs_crypto_portfolio.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# RESULTS TABLE
# =============================================================================

def print_results(smi_only: dict, with_crypto: dict):
    print("\n" + "="*65)
    print("  CRYPTO INTEGRATION — SMI vs SMI + CRYPTO")
    print("="*65)

    rows = []
    for label, res in [("SMI Only", smi_only), ("SMI + Crypto (≤15%)", with_crypto)]:
        p = res["performance"]
        rows.append({
            "Portfolio":  label,
            "Return":     f'{p["return"]:.2%}',
            "Volatility": f'{p["volatility"]:.2%}',
            "Sharpe":     f'{p["sharpe"]:.2f}',
        })
    print("\n--- Performance ---")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- Weights: SMI + Crypto portfolio ---")
    tickers = with_crypto["tickers"]
    df_w = pd.DataFrame({
        "Asset":  [ALL_TICKERS.get(t, t) for t in tickers],
        "Weight": [f'{w:.1%}' for w in with_crypto["weights"]],
        "Class":  ["Crypto" if t in CRYPTO_TICKERS else "Equity" for t in tickers]
    })
    print(df_w.to_string(index=False))
    print("="*65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Fetching data...")
    prices_all = fetch_all()
    prices_smi = fetch_smi()

    # Align SMI to same period as combined dataset
    common_start = prices_all.index[0]
    prices_smi   = prices_smi.loc[common_start:]

    returns_all = compute_returns(prices_all)
    returns_smi = compute_returns(prices_smi)

    print("Optimizing SMI-only portfolio...")
    smi_only = max_sharpe(returns_smi, max_crypto_weight=0)

    print("Optimizing SMI + Crypto portfolio (max 15% crypto)...")
    with_crypto = max_sharpe(returns_all, max_crypto_weight=0.15)

    print_results(smi_only, with_crypto)

    print("\nGenerating charts...")
    plot_correlation_matrix(returns_all)
    plot_risk_return_all_assets(returns_all)
    plot_portfolio_comparison(smi_only, with_crypto, returns_smi, returns_all)

    print("\nPart 3 complete. Charts saved in reports/figures/")