# =============================================================================
# part1_markowitz_smi.py — Markowitz Mean-Variance Optimization on SMI
# Swiss Multi-Asset Portfolio Optimizer
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    fetch_smi, compute_returns, covariance_matrix,
    annualized_return, portfolio_performance, RISK_FREE_RATE, SMI_TICKERS
)

FIGURES_PATH = "reports/figures/"
os.makedirs(FIGURES_PATH, exist_ok=True)


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def optimize_portfolio(returns: pd.DataFrame, objective: str) -> dict:
    """
    objective: 'min_vol' or 'max_sharpe'
    Returns weights + performance metrics.
    """
    n   = returns.shape[1]
    w0  = equal_weight(n)
    cov = covariance_matrix(returns).values
    mu  = annualized_return(returns).values

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * n  # long-only

    if objective == "min_vol":
        def loss(w): return np.sqrt(w @ cov @ w)

    elif objective == "max_sharpe":
        def loss(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - RISK_FREE_RATE) / vol

    result = minimize(loss, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 1000})

    weights = result.x
    perf    = portfolio_performance(weights, returns)
    return {"weights": weights, "performance": perf}


# =============================================================================
# EFFICIENT FRONTIER
# =============================================================================

def compute_efficient_frontier(returns: pd.DataFrame, n_points: int = 5000) -> pd.DataFrame:
    """Monte Carlo simulation of random portfolios."""
    n   = returns.shape[1]
    cov = covariance_matrix(returns).values
    mu  = annualized_return(returns).values

    records = []
    for _ in range(n_points):
        w      = np.random.dirichlet(np.ones(n))
        ret    = float(w @ mu)
        vol    = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - RISK_FREE_RATE) / vol
        records.append({"return": ret, "volatility": vol, "sharpe": sharpe})

    return pd.DataFrame(records)


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_efficient_frontier(frontier: pd.DataFrame,
                             portfolios: dict,
                             returns: pd.DataFrame):
    """Efficient frontier with the 3 key portfolios highlighted."""
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Scatter — random portfolios coloured by Sharpe
    sc = ax.scatter(
        frontier["volatility"], frontier["return"],
        c=frontier["sharpe"], cmap="viridis",
        alpha=0.4, s=8, zorder=1
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio", pad=0.02)

    # Key portfolios
    styles = {
        "Equal-Weight": ("o", "#e74c3c", 220),
        "Min-Vol":      ("D", "#3498db", 220),
        "Max-Sharpe":   ("*", "#f1c40f", 320),
    }
    for name, data in portfolios.items():
        perf = data["performance"]
        m, c, s = styles[name]
        ax.scatter(perf["volatility"], perf["return"],
                   marker=m, color=c, s=s, zorder=5,
                   edgecolors="white", linewidths=0.6, label=name)
        ax.annotate(
            f'{name}\nR:{perf["return"]:.1%} | σ:{perf["volatility"]:.1%} | S:{perf["sharpe"]:.2f}',
            xy=(perf["volatility"], perf["return"]),
            xytext=(12, 10), textcoords="offset points",
            fontsize=7.5, color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1e2a3a", alpha=0.85)
        )

    ax.set_xlabel("Annualized Volatility", color="white", fontsize=11)
    ax.set_ylabel("Annualized Return",     color="white", fontsize=11)
    ax.set_title("Efficient Frontier — SMI Universe (2018–Present)",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1e2a3a", edgecolor="#444",
              labelcolor="white", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    path = FIGURES_PATH + "01_efficient_frontier_smi.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_weights(portfolios: dict, tickers: list):
    """Side-by-side weight allocation bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    colors = ["#e74c3c", "#3498db", "#f1c40f"]
    names  = list(portfolios.keys())
    labels = [SMI_TICKERS.get(t, t) for t in tickers]

    for ax, name, color in zip(axes, names, colors):
        ax.set_facecolor("#0d1117")
        weights = portfolios[name]["weights"] * 100
        bars    = ax.bar(labels, weights, color=color, alpha=0.85,
                         edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=8, color="white")
        ax.set_title(name, color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel("Weight (%)", color="white")
        ax.tick_params(colors="white", axis="both")
        ax.set_ylim(0, 100)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        perf = portfolios[name]["performance"]
        ax.set_xlabel(
            f'R: {perf["return"]:.1%}  σ: {perf["volatility"]:.1%}  S: {perf["sharpe"]:.2f}',
            color="#aaaaaa", fontsize=8
        )

    plt.suptitle("Portfolio Weight Allocation — SMI",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_PATH + "02_weights_smi.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_cumulative_returns(prices: pd.DataFrame, portfolios: dict):
    """Growth of $1 invested for each portfolio strategy."""
    returns = compute_returns(prices)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    styles = {
        "Equal-Weight": ("#e74c3c", "--"),
        "Min-Vol":      ("#3498db", "-."),
        "Max-Sharpe":   ("#f1c40f", "-"),
    }
    for name, data in portfolios.items():
        w            = data["weights"]
        port_returns = returns.values @ w
        cumulative   = (1 + port_returns).cumprod()
        color, ls    = styles[name]
        ax.plot(returns.index, cumulative,
                label=name, color=color, linestyle=ls, linewidth=1.8)

    ax.set_title("Growth of $1 Invested — SMI Portfolios (2018–Present)",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Date",       color="white")
    ax.set_ylabel("Portfolio Value ($)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2a3a", edgecolor="#444",
              labelcolor="white", fontsize=9)
    ax.axhline(1, color="#555", linestyle=":", linewidth=1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    path = FIGURES_PATH + "03_cumulative_returns_smi.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# RESULTS TABLE
# =============================================================================

def print_results(portfolios: dict, tickers: list):
    labels = [SMI_TICKERS.get(t, t) for t in tickers]
    print("\n" + "="*65)
    print("  MARKOWITZ OPTIMIZATION — SMI UNIVERSE")
    print("="*65)

    # Performance table
    rows = []
    for name, data in portfolios.items():
        p = data["performance"]
        rows.append({
            "Portfolio":  name,
            "Return":     f'{p["return"]:.2%}',
            "Volatility": f'{p["volatility"]:.2%}',
            "Sharpe":     f'{p["sharpe"]:.2f}',
        })
    print("\n--- Performance ---")
    print(pd.DataFrame(rows).to_string(index=False))

    # Weights table
    print("\n--- Optimal Weights ---")
    weight_rows = {name: (data["weights"] * 100).round(1)
                   for name, data in portfolios.items()}
    df_w = pd.DataFrame(weight_rows, index=labels)
    df_w.index.name = "Asset"
    print(df_w.to_string())
    print("="*65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Fetching SMI data...")
    prices  = fetch_smi()
    returns = compute_returns(prices)
    tickers = list(prices.columns)

    print("Building portfolios...")
    eq   = equal_weight(len(tickers))
    portfolios = {
        "Equal-Weight": {"weights": eq,
                         "performance": portfolio_performance(eq, returns)},
        "Min-Vol":      optimize_portfolio(returns, "min_vol"),
        "Max-Sharpe":   optimize_portfolio(returns, "max_sharpe"),
    }

    print_results(portfolios, tickers)

    print("\nGenerating charts...")
    frontier = compute_efficient_frontier(returns)
    plot_efficient_frontier(frontier, portfolios, returns)
    plot_weights(portfolios, tickers)
    plot_cumulative_returns(prices, portfolios)

    print("\nPart 1 complete. Charts saved in reports/figures/")