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
# FUNDAMENTAL VIEWS (Black-Litterman inputs)
# =============================================================================

# Hardcoded fallback views — used when fundamental data is unavailable.
# Format: { ticker: (expected_annual_return, confidence_0_to_1) }
FALLBACK_VIEWS = {
    "ABBN.SW": (0.18, 0.70),   # ABB:      18% expected, 70% confidence
    "NOVN.SW": (0.15, 0.65),   # Novartis: 15% expected, 65% confidence
    "NESN.SW": (0.03, 0.60),   # Nestlé:    3% expected, 60% confidence
}


def get_fundamental_views(tickers: list = None,
                          confidence_base: float = 0.60) -> dict:
    """
    Derive Black-Litterman views from fundamental data.

    Macro context (risk-free rate, inflation) is sourced from swiss-finance-data
    (SNB policy rate, Swiss CPI). Per-stock fundamentals (P/E, ROE, earnings
    growth) come from yfinance, which already provides this for SMI tickers.

    The composite fundamental score maps to an expected return view:
        view_return = rf_rate + equity_risk_premium + fundamental_adjustment

    Falls back to FALLBACK_VIEWS if too little data is available.

    Parameters
    ----------
    tickers          : list of Yahoo Finance tickers (defaults to SMI_TICKERS)
    confidence_base  : baseline confidence level before completeness adjustment

    Returns
    -------
    dict : { ticker: (expected_return, confidence) }
    """
    if tickers is None:
        tickers = list(SMI_TICKERS.keys())

    # ------------------------------------------------------------------
    # Step 1: Swiss macro context via swiss-finance-data
    # ------------------------------------------------------------------
    rf_rate = RISK_FREE_RATE      # default fallback
    inflation_adj = 0.0

    try:
        import swiss_finance
        snb = swiss_finance.SNB()
        raw_rate = snb.get_policy_rate()
        # SNB returns rate in percent form (e.g. 0.25 means 0.25%)
        if isinstance(raw_rate, (int, float)) and not np.isnan(raw_rate):
            rf_rate = float(raw_rate) / 100
            rf_rate = max(0.0, min(rf_rate, 0.10))
            print(f"[fundamental_views] SNB policy rate: {rf_rate:.2%}")

        cpi = swiss_finance.CPI()
        inf_df = cpi.get_inflation_yoy()
        if inf_df is not None and not inf_df.empty:
            latest_inf = float(inf_df.iloc[-1].values[0]) / 100   # percent → decimal
            inflation_adj = max(-0.02, min(latest_inf, 0.05))
            print(f"[fundamental_views] Swiss CPI YoY: {inflation_adj:.2%}")

    except Exception as e:
        print(f"[fundamental_views] swiss-finance-data unavailable ({e}), "
              f"using default rf={rf_rate:.2%}")

    # ------------------------------------------------------------------
    # Step 2: Per-stock fundamentals via yfinance
    # ------------------------------------------------------------------
    EQUITY_RISK_PREMIUM = 0.055   # 5.5% Swiss equity risk premium
    base_return = rf_rate + EQUITY_RISK_PREMIUM

    raw_fundamentals = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            raw_fundamentals[ticker] = {
                "pe":               info.get("trailingPE"),
                "roe":              info.get("returnOnEquity"),
                "earnings_growth":  info.get("earningsGrowth"),
            }
        except Exception as e:
            print(f"[fundamental_views] Could not fetch {ticker}: {e}")
            raw_fundamentals[ticker] = {}

    # ------------------------------------------------------------------
    # Step 3: Cross-sectional scoring and view derivation
    # ------------------------------------------------------------------
    def _valid(d, key):
        v = d.get(key)
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    valid_pe  = {t: d["pe"]             for t, d in raw_fundamentals.items()
                 if _valid(d, "pe") and 0 < d["pe"] < 200}
    valid_roe = {t: d["roe"]            for t, d in raw_fundamentals.items()
                 if _valid(d, "roe")}
    valid_eg  = {t: d["earnings_growth"] for t, d in raw_fundamentals.items()
                 if _valid(d, "earnings_growth")}

    if not (valid_pe or valid_roe or valid_eg):
        print("[fundamental_views] No fundamental data available — "
              "using fallback views")
        return dict(FALLBACK_VIEWS)

    def _cross_section_score(values: dict, ticker: str, invert: bool = False) -> float:
        """Normalize a metric cross-sectionally to [-1, 1]."""
        if ticker not in values or len(values) < 2:
            return None
        vals = list(values.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return 0.0
        score = (values[ticker] - lo) / (hi - lo) * 2 - 1
        return -score if invert else score

    views = {}
    for ticker in tickers:
        components = []   # (score, weight)

        # P/E: lower is better (value signal) → invert
        pe_score = _cross_section_score(valid_pe, ticker, invert=True)
        if pe_score is not None:
            components.append((pe_score, 0.30))

        # ROE: higher is better (quality signal)
        roe_score = _cross_section_score(valid_roe, ticker, invert=False)
        if roe_score is not None:
            components.append((roe_score, 0.35))

        # Earnings growth: higher is better (momentum signal)
        eg_score = _cross_section_score(valid_eg, ticker, invert=False)
        if eg_score is not None:
            components.append((eg_score, 0.35))

        if not components:
            continue

        total_weight = sum(w for _, w in components)
        composite = sum(s * w for s, w in components) / total_weight

        # Map [-1, 1] composite score to a ±7% return adjustment
        adjustment = composite * 0.07
        view_return = round(
            max(0.01, min(base_return + adjustment + inflation_adj, 0.35)), 4
        )

        # Confidence scales with data completeness (up to +0.10 bonus)
        completeness = total_weight / (0.30 + 0.35 + 0.35)
        confidence = round(min(confidence_base + 0.10 * completeness, 0.80), 2)

        views[ticker] = (view_return, confidence)

        # Diagnostic print
        d = raw_fundamentals.get(ticker, {})
        pe_str  = f"P/E={d['pe']:.1f}"   if _valid(d, "pe")              else "P/E=N/A"
        roe_str = f"ROE={d['roe']:.1%}"  if _valid(d, "roe")             else "ROE=N/A"
        eg_str  = (f"EG={d['earnings_growth']:.1%}"
                   if _valid(d, "earnings_growth") else "EG=N/A")
        name = SMI_TICKERS.get(ticker, ticker)
        print(f"  {name:10s} [{pe_str}, {roe_str}, {eg_str}]"
              f" → view {view_return:.1%} (conf {confidence:.0%})")

    if not views:
        print("[fundamental_views] No views derived — using fallback views")
        return dict(FALLBACK_VIEWS)

    return views


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