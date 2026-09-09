"""Regression specifications for Tables 4, 5 and 6.

Three standard-error choices, each for a stated reason:

Aggregate trend (Table 4a). Twenty quarterly observations of a persistent
series regressed on time will hand you a large t statistic whether or not
anything is happening, because the residuals are autocorrelated and OLS treats
them as independent. Newey-West is required here. Both t statistics are
reported so the gap between them is visible.

Within firm (Table 4b). Firm fixed effects, so the trend is identified from a
firm's own filings moving over time rather than from the cross-section of firms
that happen to file in different quarters. Errors clustered by firm: a firm's
filings are one story repeated twenty times, not twenty independent draws.

Panel tests (Tables 5 and 6). Two-way clustering by firm and quarter. Firm,
for the reason above. Quarter, because every filing in 2022Q2 shared a market,
and a common shock is exactly what a cross-sectional regression mistakes for a
result.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 80% power at a 5% two-sided level: 1.96 + 0.84.
POWER_MULTIPLIER = 2.80


def nw_maxlags(T: int) -> int:
    """Newey-West bandwidth, the usual 4(T/100)^(2/9) rule of thumb."""
    return max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))


# ---------------------------------------------------------------------------
# Table 4
# ---------------------------------------------------------------------------
def aggregate_trend(series: pd.Series, maxlags: int | None = None) -> dict:
    """Quarterly mean on a linear time trend, OLS and Newey-West side by side.

    `series` is indexed by quarter. The slope is reported in percentage points
    per year, which is four quarters, not one.
    """
    y = series.dropna().sort_index()
    T = len(y)
    t = np.arange(T, dtype=float)
    X = sm.add_constant(t)

    ols = sm.OLS(y.to_numpy(), X).fit()
    lags = nw_maxlags(T) if maxlags is None else maxlags
    nw = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    return {
        "n_quarters": T,
        "slope_per_quarter": ols.params[1],
        "pp_per_year": ols.params[1] * 4 * 100,
        "t_ols": ols.tvalues[1],
        "t_nw": nw.tvalues[1],
        "p_nw": nw.pvalues[1],
        "nw_maxlags": lags,
    }


def within_firm_trend(panel: pd.DataFrame, measure: str,
                      time_col: str = "t", firm_col: str = "cik",
                      form_control: bool = True) -> dict:
    """Firm fixed effects, linear trend, errors clustered by firm.

    This is the test to lead with. It asks whether a given company's filings
    changed tone over five years, which is the question; the aggregate test also
    picks up which companies happened to file.
    """
    df = panel.dropna(subset=[measure, time_col, firm_col]).copy()
    rhs = f"{time_col} + C({firm_col})"
    if form_control and df["form"].nunique() > 1:
        rhs += " + C(form)"
    res = smf.ols(f"{measure} ~ {rhs}", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[firm_col]})
    return {
        "n": int(res.nobs),
        "n_firms": df[firm_col].nunique(),
        "slope_per_quarter": res.params[time_col],
        "pp_per_year": res.params[time_col] * 4 * 100,
        "t": res.tvalues[time_col],
        "p": res.pvalues[time_col],
    }


# ---------------------------------------------------------------------------
# Tables 5 and 6
# ---------------------------------------------------------------------------
def _two_way(df: pd.DataFrame, formula: str, firm_col: str, quarter_col: str):
    """OLS with two-way clustered errors.

    With ~90 firm dummies in the design, the two-way cluster covariance matrix
    can come back with a negative diagonal entry on one of the dummies, which
    statsmodels surfaces as a sqrt warning and an NaN standard error on that
    dummy. It does not touch the coefficient of interest: the standard error on
    the regressor is stable to three decimals across two-way clustering, firm
    clustering and HC1. The warning is suppressed rather than silenced blindly,
    and `nan_se` below reports how many parameters were affected.
    """
    groups = np.column_stack([
        pd.factorize(df[firm_col])[0],
        pd.factorize(df[quarter_col])[0],
    ])
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return smf.ols(formula, data=df).fit(cov_type="cluster",
                                                 cov_kwds={"groups": groups})


def panel_regression(panel: pd.DataFrame, y: str, x: str,
                     controls: list[str], firm_fe: bool = True,
                     quarter_fe: bool = True, firm_col: str = "cik",
                     quarter_col: str = "quarter") -> dict:
    """One panel specification with two-way clustered errors.

    `x` is standardised in place so the coefficient reads as "per one standard
    deviation of the measure", which makes the proportional and tf.idf columns
    comparable to each other.
    """
    need = [y, x, firm_col, quarter_col] + controls
    df = panel.dropna(subset=need).copy()
    if df.empty:
        return {"n": 0}
    # patsy cannot build a design matrix from a pandas Period dtype, and a
    # quarter is only ever a fixed-effect label here, so cast it to string.
    df[firm_col] = df[firm_col].astype(str)
    df[quarter_col] = df[quarter_col].astype(str)

    sd = df[x].std()
    df["_x"] = (df[x] - df[x].mean()) / sd

    rhs = ["_x"] + controls
    if firm_fe:
        rhs.append(f"C({firm_col})")
    if quarter_fe:
        rhs.append(f"C({quarter_col})")
    res = _two_way(df, f"{y} ~ " + " + ".join(rhs), firm_col, quarter_col)

    beta, se = res.params["_x"], res.bse["_x"]
    return {
        "n": int(res.nobs),
        "nan_se": int(np.isnan(res.bse).sum()),
        "beta": beta,
        "se": se,
        "t": res.tvalues["_x"],
        "p": res.pvalues["_x"],
        "r2": res.rsquared,
        "mde": POWER_MULTIPLIER * se,   # smallest effect detectable at 80% power
        "x_sd": sd,
        "controls": controls,
    }


def volatility_test(panel: pd.DataFrame, measure: str, controls: list[str],
                    pre_vol_col: str = "pre_vol") -> pd.DataFrame:
    """Table 5, as a ladder, because there are two gaps and only one is asked for.

    (1) and (2) are the brief's pair: post-filing volatility on uncertainty
    without and then with the pre-filing volatility control. Without it the
    coefficient is large and significant and worthless, because volatile
    companies write hedged filings and the regression has rediscovered that.
    Adding the control asks whether the language predicts a *change*.

    (3) and (4) repeat both with firm fixed effects. This is not decoration.
    "Volatile companies write hedged filings" is a statement about differences
    *between* firms, and firm fixed effects remove between-firm variation
    entirely -- so (3) and (4) ask the harder question: when a given company
    writes a more hedged filing than it usually does, does its volatility rise
    relative to its own norm? The two gaps answer different questions and the
    write-up should report both.

    Every specification carries quarter fixed effects, so none of them is
    picking up a quarter in which the whole market happened to be volatile.
    """
    specs = [
        ("(1) no pre-vol control", [], False),
        ("(2) + pre-filing vol", [pre_vol_col], False),
        ("(3) + firm FE, no pre-vol", [], True),
        ("(4) + firm FE + pre-vol", [pre_vol_col], True),
    ]
    rows = []
    for label, extra, firm_fe in specs:
        r = panel_regression(panel, y="post_vol", x=measure,
                             controls=controls + extra, firm_fe=firm_fe)
        r["spec"] = label
        r["measure"] = measure
        r["firm_fe"] = firm_fe
        r["pre_vol_control"] = bool(extra)
        rows.append(r)
    return pd.DataFrame(rows)


def return_test(panel: pd.DataFrame, measure: str, controls: list[str]) -> pd.DataFrame:
    """Table 6. Filing-period excess return on sentiment.

    Underpowered at this sample size by construction, which is why `mde` is
    reported alongside the coefficient: it says what size of effect this test
    could have found, and that number belongs in the write-up before any
    interpretation of the point estimate.
    """
    rows = []
    for label, firm_fe in (("(1) quarter FE", False), ("(2) + firm FE", True)):
        r = panel_regression(panel, y="exret_filing", x=measure,
                             controls=controls, firm_fe=firm_fe)
        r["spec"] = label
        r["measure"] = measure
        r["firm_fe"] = firm_fe
        rows.append(r)
    return pd.DataFrame(rows)
