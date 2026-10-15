"""H4: move the defensive books' bond weight into cash when bonds stop hedging equities.

Rule (docs/preregistration.md, H4): when the 63-day correlation of SPY and IEF
daily returns at the close of day t is above 0, risk parity and safe haven hold
cash earning the daily rf instead of TLT, IEF and LQD. Otherwise nothing changes.
"""
import pandas as pd

CASH = "CASH"
BONDS = ("TLT", "IEF", "LQD")
DEFENSIVE_BOOKS = ("risk_parity", "safe_haven")
CORR_WINDOW = 63


def with_cash(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Asset returns plus a cash column earning the daily risk-free rate."""
    out = returns.copy()
    out[CASH] = rf.reindex(returns.index).ffill().fillna(0.0)
    return out


def stock_bond_correlation(returns: pd.DataFrame, window: int = CORR_WINDOW) -> pd.Series:
    """Rolling correlation of SPY and IEF daily returns, known at the close of each day."""
    return returns["SPY"].rolling(window).corr(returns["IEF"])


def correlation_aware_targets(targets: dict[str, pd.DataFrame], correlation: pd.Series) -> dict[str, pd.DataFrame]:
    """Copy of the optimizer targets with a cash column and the H4 rule applied.

    Days with no correlation yet (start of sample) keep the original weights.
    """
    out = {}
    for name, t in targets.items():
        t = t.copy()
        t[CASH] = 0.0
        if name in DEFENSIVE_BOOKS:
            switch = (correlation.reindex(t.index) > 0).fillna(False)
            bonds = [b for b in BONDS if b in t.columns]
            t.loc[switch, CASH] = t.loc[switch, bonds].sum(axis=1)
            t.loc[switch, bonds] = 0.0
        out[name] = t
    return out
