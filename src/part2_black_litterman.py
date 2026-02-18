# =============================================================================
# part2_black_litterman.py — Black-Litterman Model on SMI
# Swiss Multi-Asset Portfolio Optimizer
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# BLACK-LITTERMAN CORE
# =============================================================================

def compute_market_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Proxy market-cap weights using inverse volatility.
    (We don't have live market cap — this is a transparent approximation.)
    """
    vols = returns.std()
    inv_vol = 1 / vols
    return (inv_vol / inv_vol.sum()).values


def black_litterman(returns: pd.DataFrame,
                    views: dict,
                    tau: float = 0.05) -> dict:
    """
    Black-Litterman model.

    Parameters
    ----------
    returns : pd.DataFrame — historical log returns
    views   : dict — analyst views, format:
              { "asset_ticker": (view_return_float, confidence_float) }
              e.g. { "ABBN.SW": (0.18, 0.7) }
              confidence in [0, 1] — 1 = very high conviction
    tau     : float — scalar uncertainty on prior (default 0.05)

    Returns
    -------
    dict with BL expected returns, posterior cov, optimal weights
    """
    tickers = list(returns.columns)
    n       = len(tickers)
    cov     = covariance_matrix(returns).values  # Σ annualized
    w_mkt   = compute_market_weights(returns)

    # --- Prior: implied equilibrium returns (reverse optimization) ---
    # δ = risk aversion coefficient (market Sharpe / market variance proxy)
    delta   = 2.5
    pi      = delta * cov @ w_mkt   # implied returns vector

    # --- Views matrix P and vector Q ---
    view_tickers = [t for t in views if t in tickers]
    k            = len(view_tickers)

    P = np.zeros((k, n))
    Q = np.zeros(k)
    O = np.zeros((k, k))  # diagonal uncertainty matrix

    for i, ticker in enumerate(view_tickers):
        view_return, confidence = views[ticker]
        j         = tickers.index(ticker)
        P[i, j]   = 1.0
        Q[i]      = view_return
        # Omega: lower confidence = higher uncertainty
        O[i, i]   = (1 - confidence) / confidence * (tau * cov[j, j])

    # --- BL posterior expected returns ---
    tau_cov = tau * cov
    M       = np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(O) @ P
    BL_mu   = np.linalg.inv(M) @ (
        np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(O) @ Q
    )

    # --- Posterior covariance ---
    BL_cov  = cov + np.linalg.inv(M)

    # --- Optimize Max-Sharpe on BL posterior ---
    w0          = np.ones(n) / n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * n

    def neg_sharpe(w):
        ret = float(w @ BL_mu)
        vol = float(np.sqrt(w @ BL_cov @ w))
        return -(ret - RISK_FREE_RATE) / vol

    result  = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
    w_bl    = result.x

    # Performance using BL posterior
    bl_ret  = float(w_bl @ BL_mu)
    bl_vol  = float(np.sqrt(w_bl @ BL_cov @ w_bl))
    bl_sharpe = (bl_ret - RISK_FREE_RATE) / bl_vol

    return {
        "weights":        w_bl,
        "bl_returns":     BL_mu,
        "bl_cov":         BL_cov,
        "implied_returns": pi,
        "tickers":        tickers,
        "performance": {
            "return":     round(bl_ret, 4),
            "volatility": round(bl_vol, 4),
            "sharpe":     round(bl_sharpe, 4)
        }
    }


# =============================================================================
# MARKOWITZ BASELINE (for comparison)
# =============================================================================

def markowitz_max_sharpe(returns: pd.DataFrame) -> dict:
    n           = returns.shape[1]
    cov         = covariance_matrix(returns).values
    mu          = annualized_return(returns).values
    w0          = np.ones(n) / n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * n

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        return -(ret - RISK_FREE_RATE) / vol

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 1000})
    w      = result.x
    perf   = portfolio_performance(w, returns)
    return {"weights": w, "performance": perf}


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_bl_vs_markowitz(bl_result: dict,
                          mkw_result: dict,
                          returns: pd.DataFrame,
                          views: dict):
    """Side-by-side comparison: BL weights vs Markowitz weights."""
    tickers = bl_result["tickers"]
    labels  = [SMI_TICKERS.get(t, t) for t in tickers]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")

    datasets = [
        ("Markowitz Max-Sharpe\n(historical returns only)",
         mkw_result["weights"], mkw_result["performance"], "#f1c40f"),
        ("Black-Litterman Max-Sharpe\n(+ analyst views)",
         bl_result["weights"],  bl_result["performance"],  "#2ecc71"),
    ]

    for ax, (title, weights, perf, color) in zip(axes, datasets):
        ax.set_facecolor("#0d1117")
        bars = ax.bar(labels, weights * 100, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, weights * 100):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, color="white")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel("Weight (%)", color="white")
        ax.set_ylim(0, 100)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_xlabel(
            f'R: {perf["return"]:.1%}  σ: {perf["volatility"]:.1%}  S: {perf["sharpe"]:.2f}',
            color="#aaaaaa", fontsize=8
        )

    # Annotate the views used
    view_text = "Analyst Views applied:\n" + "\n".join(
        f"  {SMI_TICKERS.get(t, t)}: {r:.0%} expected  (conf. {c:.0%})"
        for t, (r, c) in views.items()
    )
    fig.text(0.5, -0.05, view_text, ha="center", color="#aaaaaa",
             fontsize=8, style="italic")

    plt.suptitle("Black-Litterman vs Markowitz — Weight Allocation (SMI)",
                 color="white", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    path = FIGURES_PATH + "04_bl_vs_markowitz_weights.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_implied_vs_bl_returns(bl_result: dict):
    """Compare implied equilibrium returns vs BL posterior returns."""
    tickers  = bl_result["tickers"]
    labels   = [SMI_TICKERS.get(t, t) for t in tickers]
    implied  = bl_result["implied_returns"] * 100
    bl_ret   = bl_result["bl_returns"] * 100

    x   = np.arange(len(labels))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.bar(x - w/2, implied, w, label="Implied (equilibrium)",
           color="#3498db", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, bl_ret,  w, label="Black-Litterman posterior",
           color="#2ecc71", alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white")
    ax.set_ylabel("Expected Return (%)", color="white")
    ax.set_title("Implied Equilibrium vs Black-Litterman Posterior Returns",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2a3a", edgecolor="#444",
              labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.axhline(0, color="#555", linewidth=0.8)

    plt.tight_layout()
    path = FIGURES_PATH + "05_implied_vs_bl_returns.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_cumulative_bl_vs_mkw(prices: pd.DataFrame,
                               bl_result: dict,
                               mkw_result: dict):
    """Cumulative return: BL portfolio vs Markowitz vs Equal-Weight."""
    returns  = compute_returns(prices)
    n        = returns.shape[1]
    eq_w     = np.ones(n) / n

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for label, weights, color, ls in [
        ("Equal-Weight",           eq_w,                  "#e74c3c", "--"),
        ("Markowitz Max-Sharpe",   mkw_result["weights"], "#f1c40f", "-."),
        ("Black-Litterman",        bl_result["weights"],  "#2ecc71", "-"),
    ]:
        cum = (1 + returns.values @ weights).cumprod()
        ax.plot(returns.index, cum, label=label,
                color=color, linestyle=ls, linewidth=1.8)

    ax.set_title("Growth of $1 — BL vs Markowitz vs Equal-Weight (SMI)",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date",             color="white")
    ax.set_ylabel("Portfolio Value ($)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2a3a", edgecolor="#444",
              labelcolor="white", fontsize=9)
    ax.axhline(1, color="#555", linestyle=":", linewidth=1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    path = FIGURES_PATH + "06_cumulative_bl_vs_mkw.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# RESULTS TABLE
# =============================================================================

def print_results(bl_result: dict, mkw_result: dict):
    tickers = bl_result["tickers"]
    labels  = [SMI_TICKERS.get(t, t) for t in tickers]

    print("\n" + "="*65)
    print("  BLACK-LITTERMAN vs MARKOWITZ — SMI UNIVERSE")
    print("="*65)

    rows = [
        {"Model": "Markowitz Max-Sharpe",
         "Return":     f'{mkw_result["performance"]["return"]:.2%}',
         "Volatility": f'{mkw_result["performance"]["volatility"]:.2%}',
         "Sharpe":     f'{mkw_result["performance"]["sharpe"]:.2f}'},
        {"Model": "Black-Litterman",
         "Return":     f'{bl_result["performance"]["return"]:.2%}',
         "Volatility": f'{bl_result["performance"]["volatility"]:.2%}',
         "Sharpe":     f'{bl_result["performance"]["sharpe"]:.2f}'},
    ]
    print("\n--- Performance ---")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- Weight Comparison ---")
    df_w = pd.DataFrame({
        "Markowitz": (mkw_result["weights"] * 100).round(1),
        "Black-Litterman": (bl_result["weights"] * 100).round(1),
    }, index=labels)
    df_w.index.name = "Asset"
    print(df_w.to_string())
    print("="*65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- Analyst views ---
    # Format: { "TICKER": (expected_annual_return, confidence_0_to_1) }
    # These are informed views based on Part 1 results:
    # ABB and Novartis showed highest Sharpe → we express bullish views
    # Nestlé showed near-zero Sharpe → mild bearish view
    VIEWS = {
        "ABBN.SW": (0.18, 0.70),   # ABB:      18% expected, 70% confidence
        "NOVN.SW": (0.15, 0.65),   # Novartis: 15% expected, 65% confidence
        "NESN.SW": (0.03, 0.60),   # Nestlé:    3% expected, 60% confidence
    }

    print("Fetching SMI data...")
    prices  = fetch_smi()
    returns = compute_returns(prices)

    print("Running Black-Litterman model...")
    bl_result  = black_litterman(returns, VIEWS)

    print("Running Markowitz baseline...")
    mkw_result = markowitz_max_sharpe(returns)

    print_results(bl_result, mkw_result)

    print("\nGenerating charts...")
    plot_bl_vs_markowitz(bl_result, mkw_result, returns, VIEWS)
    plot_implied_vs_bl_returns(bl_result)
    plot_cumulative_bl_vs_mkw(prices, bl_result, mkw_result)

    print("\nPart 2 complete. Charts saved in reports/figures/")