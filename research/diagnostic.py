"""H1: do the regime posteriors predict which optimizer will do better?"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from optimization.switcher import LABELS

# Crash is omitted because the four posteriors sum to 1.
REGRESSORS = ["Bull", "Sideways", "Bear"]


def newey_west_lag(n_obs: int) -> int:
    """Newey-West (1994) rule of thumb, floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def rotation_regressions(book_returns: dict[str, pd.Series], posteriors: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Regress each book's next-day return minus the books' average on the day-t posteriors.

    One row per optimizer: slopes, HAC Wald test that all slopes are zero, and
    Holm-adjusted p-values across optimizers.
    """
    books = pd.DataFrame(book_returns).dropna()
    relative = books.sub(books.mean(axis=1), axis=0)
    lagged = posteriors[LABELS].shift(1).reindex(relative.index)
    data = relative.join(lagged).dropna()
    x = sm.add_constant(data[REGRESSORS])
    lag = newey_west_lag(len(data))

    rows = {}
    for name in books.columns:
        fit = sm.OLS(data[name], x).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        wald = fit.wald_test(np.eye(len(x.columns))[1:], scalar=True)
        rows[name] = {
            **{f"slope_{r}": fit.params[r] for r in REGRESSORS},
            **{f"t_{r}": fit.tvalues[r] for r in REGRESSORS},
            "wald_stat": float(wald.statistic),
            "p_value": float(wald.pvalue),
            "n_obs": len(data),
            "nw_lag": lag,
        }
    table = pd.DataFrame(rows).T
    reject, p_holm, _, _ = multipletests(table["p_value"].astype(float), alpha=alpha, method="holm")
    table["p_holm"] = p_holm
    table["reject"] = reject
    return table
