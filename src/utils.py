# =============================================================================
# utils.py — Data fetching & shared financial metrics
# Swiss Multi-Asset Portfolio Optimizer
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

SMI_TICKERS = {
    "NESN.SW": "Nestlé",
    "NOVN.SW": "Novartis",
    "UBSG.SW": "UBS",
    "ROG.SW":  "Roche",
    "ABBN.SW": "ABB"
}

CRYPTO_TICKERS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum"
}

START_DATE = "2018-01-01"
RISK_FREE_RATE = 0.02  # 2% — Swiss National Bank approximation
TRADING_DAYS = 252


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_prices(tickers: list, start: str = START_DATE) -> pd.DataFrame:
    """
    Fetch adjusted closing prices via yfinance.
    Returns a clean DataFrame with forward-fill for missing values.
    """
    print(f"Fetching data for: {tickers}")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]

    # Handle single ticker edge case
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Forward-fill then drop any remaining NaN
    prices = raw.ffill().dropna()
    print(f"Data fetched: {prices.shape[0]} rows from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def fetch_smi(start: str = START_DATE) -> pd.DataFrame:
    """Fetch SMI equities only."""
    return fetch_prices(list(SMI_TICKERS.keys()), start=start)


def fetch_crypto(start: str = START_DATE) -> pd.DataFrame:
    """Fetch crypto assets only."""
    return fetch_prices(list(CRYPTO_TICKERS.keys()), start=start)


def fetch_all(start: str = START_DATE) -> pd.DataFrame:
    """Fetch SMI + crypto combined, aligned on common trading days."""
    smi    = fetch_smi(start)
    crypto = fetch_crypto(start)

    # Align on common dates (inner join)
    combined = smi.join(crypto, how="inner")
    print(f"Combined dataset: {combined.shape[1]} assets, {combined.shape[0]} trading days")
    return combined


# =============================================================================
# FINANCIAL METRICS
# =============================================================================

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(returns: pd.DataFrame) -> pd.Series:
    """Mean log return annualized."""
    return returns.mean() * TRADING_DAYS


def annualized_volatility(returns: pd.DataFrame) -> pd.Series:
    """Annualized standard deviation."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.DataFrame) -> pd.Series:
    """Sharpe ratio with Swiss risk-free rate."""
    excess = annualized_return(returns) - RISK_FREE_RATE
    return excess / annualized_volatility(returns)


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance matrix."""
    return returns.cov() * TRADING_DAYS


def portfolio_performance(weights: np.ndarray, returns: pd.DataFrame) -> dict:
    """
    Compute portfolio-level metrics given weights.
    weights : array aligned with returns.columns
    """
    w   = np.array(weights)
    cov = covariance_matrix(returns).values
    mu  = annualized_return(returns).values

    port_return = float(w @ mu)
    port_vol    = float(np.sqrt(w @ cov @ w))
    port_sharpe = (port_return - RISK_FREE_RATE) / port_vol

    return {
        "return":     round(port_return, 4),
        "volatility": round(port_vol, 4),
        "sharpe":     round(port_sharpe, 4)
    }


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    prices  = fetch_smi()
    returns = compute_returns(prices)

    print("\n--- Annualized Metrics (SMI) ---")
    metrics = pd.DataFrame({
        "Return":     annualized_return(returns).map("{:.2%}".format),
        "Volatility": annualized_volatility(returns).map("{:.2%}".format),
        "Sharpe":     sharpe_ratio(returns).map("{:.2f}".format)
    })
    print(metrics.to_string())