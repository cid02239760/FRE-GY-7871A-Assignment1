"""The trading calendar, the day-0 rule, and the event windows.

None of this is in the starter code. The pieces here are the ones where a
plausible-looking shortcut quietly produces a look-ahead bug:

Day 0. EDGAR's filing date is not automatically tradable. A 10-K accepted at
21:19 UTC is accepted at 16:19 New York time, after the close, and the first
price that can reflect it is the next session. So day 0 is the first trading day
on or after the later of (a) the filing date and (b) the acceptance date, pushed
forward one day when acceptance lands at or after 16:00 Eastern. The count of
filings this moves is reported in the write-up.

Share counts. Market cap uses the share count printed on the filing being
scored, from data/prices/shares.csv, never today's count. Filings with no share
fact are left missing and counted, not filled forward from a later filing.

Windows, per the brief:
    filing-period excess return   [0, +3] against SPY, from the close on day -1
    pre-filing realised vol       [-60, -6], annualised
    post-filing realised vol      [+4, +63], annualised
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Window bounds in trading days relative to day 0, inclusive on both ends.
RET_WINDOW = (0, 3)
PRE_VOL_WINDOW = (-60, -6)
POST_VOL_WINDOW = (4, 63)
PRE_RET_WINDOW = (-60, -6)

MIN_PRICE = 3.0                 # price on day -1
MIN_WORDS = {"10-K": 2000, "10-Q": 1000}
MIN_PRE_DAYS = 60               # trading days of returns required either side
MIN_POST_DAYS = 60


# ---------------------------------------------------------------------------
# Calendar and day 0
# ---------------------------------------------------------------------------
def trading_calendar(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DatetimeIndex:
    """Sessions on which the benchmark traded. Our definition of a trading day.

    Using the benchmark's own index rather than a holiday library keeps the
    calendar consistent with the prices the returns are computed from, and adds
    no dependency beyond requirements.txt.
    """
    if benchmark not in prices.columns:
        raise KeyError(f"{benchmark} not in prices; cannot build a calendar")
    return pd.DatetimeIndex(prices.index[prices[benchmark].notna()]).sort_values()


def _first_session_on_or_after(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Map each date to the first session on or after it; NaT past the calendar."""
    pos = calendar.searchsorted(pd.DatetimeIndex(dates), side="left")
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = pos < len(calendar)
    out.loc[ok] = calendar[pos[ok]]
    return out


def assign_day0(meta: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Add day0, day0_naive and day0_moved to `meta`.

    `day0_naive` is what you get by treating the EDGAR filing date as tradable.
    The difference between the two columns is the point-in-time correction, and
    its frequency is a reported number.
    """
    out = meta.copy()
    acc_utc = pd.to_datetime(out["acceptance_datetime"], utc=True, errors="coerce")
    acc_et = acc_utc.dt.tz_convert("America/New_York")

    # "at or after 16:00 Eastern" -> the filing was not tradable that session
    late = (acc_et.dt.hour >= 16).fillna(False)
    acc_date = acc_et.dt.tz_localize(None).dt.normalize()
    eff_acc = acc_date + pd.to_timedelta(late.astype(int), unit="D")

    filing = pd.to_datetime(out["filing_date"]).dt.normalize()
    # Missing acceptance stamps fall back to the filing date rather than to NaT,
    # so a filing is never dropped for want of a timestamp we could do without.
    candidate = pd.concat([filing, eff_acc], axis=1).max(axis=1)

    out["acceptance_et"] = acc_et
    out["accepted_after_close"] = late
    out["day0"] = _first_session_on_or_after(candidate, calendar)
    out["day0_naive"] = _first_session_on_or_after(filing, calendar)
    out["day0_moved"] = out["day0"].ne(out["day0_naive"]) & out["day0"].notna()
    return out


# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------
def _annualised_vol(rets: np.ndarray) -> float:
    if rets.size < 2 or np.isnan(rets).all():
        return np.nan
    return float(np.nanstd(rets, ddof=1) * np.sqrt(TRADING_DAYS))


def _buy_and_hold(px: np.ndarray, i_from: int, i_to: int) -> float:
    """Return from the close at i_from to the close at i_to."""
    if i_from < 0 or i_to >= px.size:
        return np.nan
    p0, p1 = px[i_from], px[i_to]
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
        return np.nan
    return float(p1 / p0 - 1.0)


def build_event_panel(meta: pd.DataFrame, prices: pd.DataFrame,
                      volume: pd.DataFrame, calendar: pd.DatetimeIndex,
                      benchmark: str = "SPY") -> pd.DataFrame:
    """One row per filing with returns, volatilities and the market controls.

    Everything is positional in the trading calendar, so "day -60" means sixty
    sessions back, not sixty calendar days.
    """
    px_all = prices.reindex(calendar)
    vol_all = volume.reindex(calendar)

    bench = px_all[benchmark].to_numpy(dtype=float)
    bench_lr = np.diff(np.log(bench))          # aligned to calendar[1:]

    day0_pos = pd.Series(calendar.searchsorted(pd.DatetimeIndex(meta["day0"]), side="left"),
                         index=meta.index)

    rows = []
    for ticker, grp in meta.groupby("ticker", sort=False):
        if ticker not in px_all.columns:
            for idx in grp.index:
                rows.append({"_idx": idx, "has_prices": False})
            continue

        px = px_all[ticker].to_numpy(dtype=float)
        vl = vol_all[ticker].to_numpy(dtype=float) if ticker in vol_all.columns \
            else np.full_like(px, np.nan)
        lr = np.diff(np.log(np.where(px > 0, px, np.nan)))

        for idx in grp.index:
            t0 = int(day0_pos.loc[idx])
            r = {"_idx": idx, "has_prices": True}
            if pd.isna(meta.loc[idx, "day0"]) or t0 >= len(calendar):
                rows.append(r)
                continue

            # -- day -1, the base for every return in the event window --------
            r["price_m1"] = px[t0 - 1] if t0 - 1 >= 0 else np.nan

            # -- filing-period excess return, [0,+3] from the close on -1 -----
            a, b = RET_WINDOW
            r_stock = _buy_and_hold(px, t0 - 1, t0 + b)
            r_bench = _buy_and_hold(bench, t0 - 1, t0 + b)
            r["ret_filing"] = r_stock
            r["exret_filing"] = (r_stock - r_bench
                                 if np.isfinite(r_stock) and np.isfinite(r_bench) else np.nan)

            # -- realised volatility, before and after ------------------------
            # lr[k] is the return from calendar[k] to calendar[k+1], so the
            # returns *on* sessions s..e are lr[s-1 : e]. Note [-60,-6] is 55
            # returns, not 60: the window is inclusive on both ends.
            for name, (s, e) in (("pre_vol", PRE_VOL_WINDOW), ("post_vol", POST_VOL_WINDOW)):
                lo, hi = t0 + s - 1, t0 + e
                r[name] = (_annualised_vol(lr[lo:hi])
                           if lo >= 0 and hi <= lr.size else np.nan)

            # -- return history available either side of day 0 ----------------
            # This is filter 5, and it is a data-sufficiency test, separate from
            # the volatility windows above: does this ticker actually trade
            # either side of the filing, or is it a recent listing / a stub?
            for name, (lo, hi) in (("n_ret_pre", (t0 - MIN_PRE_DAYS, t0)),
                                   ("n_ret_post", (t0, t0 + MIN_POST_DAYS))):
                seg = lr[max(lo, 0):min(hi, lr.size)] if hi > 0 else np.array([])
                r[name] = int(np.isfinite(seg).sum())

            # -- pre-filing excess return and dollar volume controls ----------
            s, e = PRE_RET_WINDOW
            pre_s, pre_e = _buy_and_hold(px, t0 + s - 1, t0 + e), _buy_and_hold(bench, t0 + s - 1, t0 + e)
            r["pre_exret"] = (pre_s - pre_e
                              if np.isfinite(pre_s) and np.isfinite(pre_e) else np.nan)

            lo, hi = t0 + s, t0 + e + 1
            if lo >= 0 and hi <= px.size:
                dollar = px[lo:hi] * vl[lo:hi]
                r["dollar_volume"] = float(np.nanmean(dollar)) if np.isfinite(dollar).any() else np.nan
            else:
                r["dollar_volume"] = np.nan

            rows.append(r)

    ev = pd.DataFrame(rows).set_index("_idx").reindex(meta.index)
    return meta.join(ev)


def attach_shares(panel: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time market cap from the share count printed on each filing.

    Joined on accession, so the count comes from the document being scored. No
    forward filling: a filing with no share fact keeps a missing market cap and
    is counted in the write-up.
    """
    s = shares[["accession", "shares_outstanding", "shares_tag"]].drop_duplicates("accession")
    out = panel.merge(s, on="accession", how="left")
    out["mktcap"] = out["price_m1"] * out["shares_outstanding"]
    return out


# ---------------------------------------------------------------------------
# Sample filters -> Table 1
# ---------------------------------------------------------------------------
def count_amendments(universe: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """How many 10-K/A and 10-Q/A the universe filed in the window.

    src/edgar.py excludes amendments at the query (`include_amendments=False`),
    so they never reach filings_meta.csv and filter 1 would otherwise report a
    zero that only means "we never looked". This re-queries with amendments on
    and counts them, so Table 1's first row is a real number.

    Cheap: the submissions payloads are already in the EdgarClient cache.
    """
    from .config import FORMS, SEC_USER_AGENT
    from .edgar import EdgarClient

    client = EdgarClient(SEC_USER_AGENT or None)
    rows = []
    for _, firm in universe.iterrows():
        try:
            f = client.list_filings(firm["cik"], FORMS, start, end, include_amendments=True)
        except Exception:  # noqa: BLE001
            continue
        if len(f):
            amd = f[f["form"].str.contains("/A")]
            for form, n in amd["form"].value_counts().items():
                rows.append({"ticker": firm["ticker"], "form": form, "n": n})
    return pd.DataFrame(rows)



def apply_filters(panel: pd.DataFrame, amendments_dropped: int = 0
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The five filters, in the brief's order, counting what each one removes.

    Returns the surviving panel and the Table 1 waterfall.
    """
    steps = []
    df = panel.copy()

    def record(label: str, keep: pd.Series) -> pd.DataFrame:
        nonlocal df
        before = len(df)
        df = df[keep].copy()
        steps.append({"filter": label, "removed": before - len(df), "remaining": len(df)})
        return df

    steps.append({"filter": "10-K and 10-Q filings, 2021-2025, 93 ARK filers",
                  "removed": 0, "remaining": len(df)})

    # 1. amendments and parse failures. Amendments never enter filings_meta.csv
    #    (src/edgar.py excludes them at the query), so their count is passed in
    #    from a separate query rather than invented here.
    if amendments_dropped:
        steps.append({"filter": "1a. Amendments (10-K/A, 10-Q/A), excluded at query",
                      "removed": amendments_dropped, "remaining": len(df)})
    record("1b. Failed to parse (zero words)", df["n_words"].fillna(0) > 0)

    # 2. length floors
    floor = df["form"].map(MIN_WORDS).fillna(0)
    record("2. Below word floor (2,000 for 10-K, 1,000 for 10-Q)", df["n_words"] >= floor)

    # 3. one filing per company per calendar quarter, earliest kept.
    #    Split in two because the causes are different and only the second is
    #    what the brief's filter is really about. Alphabet is in the ARK
    #    universe under both GOOG and GOOGL: one CIK, one set of filings,
    #    counted twice. Sorting on ticker makes the survivor deterministic
    #    rather than dependent on row order.
    df = df.sort_values(["cik", "day0", "filing_date", "ticker"])
    record("3a. Dual-class listing (same filing under two tickers)",
           ~df.duplicated(subset=["accession"], keep="first"))

    q = pd.to_datetime(df["filing_date"]).dt.to_period("Q")
    record("3b. Multiple filings per company-quarter (earliest kept)",
           ~pd.Series(df.set_index([df["cik"], q]).index.duplicated(keep="first"),
                      index=df.index))

    # 4. usable day 0 and a real price behind it
    record("4a. No usable day 0", df["day0"].notna() & df["has_prices"].fillna(False))
    record("4b. Price on day -1 below $3", df["price_m1"] >= MIN_PRICE)

    # 5. enough return history either side
    record("5. Fewer than 60 trading days of returns either side",
           (df["n_ret_pre"] >= MIN_PRE_DAYS) & (df["n_ret_post"] >= MIN_POST_DAYS))

    return df, pd.DataFrame(steps)
