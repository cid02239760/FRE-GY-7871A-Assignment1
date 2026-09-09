"""Table and figure construction.

Everything here writes to outputs/ as CSV or PNG so the notebook stays a
narrative rather than a wall of formatting code.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEASURES = {
    "neg_prop": "Fin-Neg, proportional",
    "unc_prop": "Fin-Unc, proportional",
    "neg_tfidf": "Fin-Neg, tf.idf",
    "unc_tfidf": "Fin-Unc, tf.idf",
}


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------
def table2_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for both measures, 10-K and 10-Q separately."""
    rows = []
    for form in ["10-K", "10-Q", "All"]:
        sub = panel if form == "All" else panel[panel["form"] == form]
        for col, label in MEASURES.items():
            s = sub[col].dropna()
            rows.append({
                "form": form, "measure": label, "n": len(s),
                "mean": s.mean(), "sd": s.std(),
                "p10": s.quantile(0.10), "median": s.median(), "p90": s.quantile(0.90),
            })
    return pd.DataFrame(rows)


def measure_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Sentiment against uncertainty, within each weighting scheme.

    Q2 turns on these two numbers: if they are high, much of the rest of the
    report is one result reported twice, and that has to be said plainly.
    """
    rows = []
    for form in ["10-K", "10-Q", "All"]:
        sub = panel if form == "All" else panel[panel["form"] == form]
        rows.append({
            "form": form,
            "corr_proportional": sub["neg_prop"].corr(sub["unc_prop"]),
            "corr_tfidf": sub["neg_tfidf"].corr(sub["unc_tfidf"]),
            "corr_neg_prop_vs_tfidf": sub["neg_prop"].corr(sub["neg_tfidf"]),
            "corr_unc_prop_vs_tfidf": sub["unc_prop"].corr(sub["unc_tfidf"]),
            "n": len(sub),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 3
# ---------------------------------------------------------------------------
def table3_top_words(counts: pd.DataFrame, word_lists: dict[str, set[str]],
                     accessions: set[str], top: int = 30) -> pd.DataFrame:
    """The `top` most frequent words on each list, with each word's share.

    The share is of that list's total count across the corpus, which is what
    makes the concentration in Q1 legible: if ten words carry half the count,
    the measure is those ten words.
    """
    c = counts[counts["accession"].isin(accessions)]
    rows = []
    cols = ["list", "rank", "word", "count", "share_of_list", "pct_of_filings"]
    if c.empty:
        return pd.DataFrame(columns=cols)
    for cat, short in (("Negative", "Fin-Neg"), ("Uncertainty", "Fin-Unc")):
        sub = c[c["word"].isin(word_lists[cat])]
        tot = sub.groupby("word")["count"].sum().sort_values(ascending=False)
        grand = tot.sum()
        ndocs = sub.groupby("word")["accession"].nunique()
        n_filings = c["accession"].nunique()
        for rank, (w, n) in enumerate(tot.head(top).items(), 1):
            rows.append({
                "list": short, "rank": rank, "word": w, "count": int(n),
                "share_of_list": n / grand,
                "pct_of_filings": ndocs.get(w, 0) / n_filings,
            })
    return pd.DataFrame(rows)


def concentration(counts: pd.DataFrame, word_lists: dict[str, set[str]],
                  accessions: set[str]) -> pd.DataFrame:
    """How much of each list's count sits in its top 10, 30 and 100 words."""
    c = counts[counts["accession"].isin(accessions)]
    rows = []
    for cat, short in (("Negative", "Fin-Neg"), ("Uncertainty", "Fin-Unc")):
        sub = c[c["word"].isin(word_lists[cat])]
        tot = sub.groupby("word")["count"].sum().sort_values(ascending=False)
        grand = tot.sum()
        rows.append({
            "list": short,
            "list_size": len(word_lists[cat]),
            "words_observed": len(tot),
            "top10_share": tot.head(10).sum() / grand,
            "top30_share": tot.head(30).sum() / grand,
            "top100_share": tot.head(100).sum() / grand,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
def figure1(panel: pd.DataFrame, vix: pd.Series, path,
            measures=("neg_prop", "unc_prop")) -> pd.DataFrame:
    """Both measures by quarter, split by form type, with the VIX behind them.

    Splitting by form type is not decoration. 10-Ks are longer and heavier in
    risk language and they cluster in Q1, so a pooled line carries an annual
    sawtooth that is pure calendar artefact and looks exactly like a result.
    """
    q = panel.groupby(["quarter", "form"])[list(measures)].mean().reset_index()

    # One shared x axis, defined by the filing sample. Plotting two categorical
    # series against each other would otherwise append the VIX's extra quarters
    # (the price history starts before the sample and ends after it) to the
    # right-hand end of the axis, out of order.
    quarters = sorted(panel["quarter"].unique())
    pos = {qq: i for i, qq in enumerate(quarters)}
    vq = (vix.groupby(pd.PeriodIndex(vix.index, freq="Q")).mean()
          .reindex(quarters))

    fig, axes = plt.subplots(len(measures), 1, figsize=(11, 7), sharex=True)
    colors = {"10-K": "#1f77b4", "10-Q": "#d62728"}

    for ax, m in zip(np.atleast_1d(axes), measures):
        for form, grp in q.groupby("form"):
            ax.plot(grp["quarter"].map(pos), grp[m] * 100, marker="o", ms=3.5,
                    lw=1.6, color=colors.get(form, None), label=form)
        ax.set_ylabel(f"{MEASURES[m]}\n(% of words)")
        ax.grid(alpha=0.25, lw=0.5)

        ax2 = ax.twinx()
        ax2.plot(range(len(quarters)), vq.to_numpy(dtype=float),
                 color="0.55", lw=1.2, ls="--", zorder=0, label="VIX")
        ax2.set_ylabel("VIX", color="0.45")
        ax2.tick_params(axis="y", colors="0.45")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, frameon=False, loc="upper left")

    last = np.atleast_1d(axes)[-1]
    last.set_xticks(range(len(quarters)))
    last.set_xticklabels([str(qq) for qq in quarters], rotation=90, fontsize=8)
    fig.suptitle("Figure 1. Tone by quarter, 2021-2025, by form type, with the VIX",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return q
