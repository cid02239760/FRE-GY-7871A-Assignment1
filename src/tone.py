"""Scoring filings on the Loughran-McDonald word lists.

Two weighting schemes, per step 3 of the assignment brief:

    proportional    count of list words / total words
    tf.idf          equation (1) of Loughran & McDonald (2011)

Equation (1), natural logs throughout:

    w_ij = [ (1 + ln tf_ij) / (1 + ln a_j) ] * ln( N / df_i )    if tf_ij > 0
         = 0                                                     otherwise

    a_j  = the average word count *within* document j
         = (total words in j) / (distinct words in j)

    score_j = sum of w_ij over the words i on the list being scored

`N` is the number of documents in the corpus and `df_i` the number of documents
containing word i. Both are properties of the corpus, not of a document, so they
have to be computed on the *final* sample: scoring one corpus and running the
regressions on another is a real error, not a stylistic one.

Note what the idf term does to a word that appears in every filing. df_i = N
gives ln(1) = 0, so MAY and APPROXIMATELY contribute nothing. That is the whole
reason weighting matters more for uncertainty than for sentiment, and it is why
the two schemes are reported side by side rather than one standing in for both.

The three-document self-check in the README is `selfcheck()` below, and
`tests/test_tone.py` runs it through this module's real scoring path.
"""

from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .parse import tokenize


# ---------------------------------------------------------------------------
# Building the count matrix
# ---------------------------------------------------------------------------
def count_filing(text_path: Path, vocab: set[str]) -> tuple[Counter, int, int]:
    """One filing -> (counts restricted to `vocab`, total words, distinct words).

    The totals are counted over *all* tokens, before the restriction. They are
    the denominators (n_words) and the a_j input (n_distinct), so restricting
    first would quietly corrupt both.
    """
    with gzip.open(text_path, "rt", encoding="utf-8") as fh:
        tokens = tokenize(fh.read())
    full = Counter(tokens)
    restricted = Counter({w: c for w, c in full.items() if w in vocab})
    return restricted, len(tokens), len(full)


def build_counts(meta: pd.DataFrame, vocab: set[str], root: Path,
                 progress: bool = True) -> pd.DataFrame:
    """Long count frame for every filing in `meta`: accession, word, count.

    Restricted to `vocab` (the union of the two word lists, ~2,600 words) because
    that is all the scores, the document frequencies and Table 3 ever need. The
    full vocabulary would be ~10x larger for no gain.
    """
    rows = []
    total = len(meta)
    for i, (_, f) in enumerate(meta.iterrows(), 1):
        path = root / f["text_path"]
        try:
            counts, _, _ = count_filing(path, vocab)
        except (OSError, EOFError) as exc:
            print(f"  ! unreadable {f['accession']}: {exc}")
            continue
        for word, c in counts.items():
            rows.append((f["accession"], word, c))
        if progress and i % 200 == 0:
            print(f"  counted {i}/{total} filings")
    return pd.DataFrame(rows, columns=["accession", "word", "count"])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_proportional(counts: pd.DataFrame, docs: pd.DataFrame,
                       word_list: set[str]) -> pd.Series:
    """Share of a document's words that are on `word_list`.

    `docs` is indexed by document id and carries n_words and n_distinct.
    """
    sub = counts[counts["word"].isin(word_list)]
    hits = sub.groupby("accession")["count"].sum()
    hits = hits.reindex(docs.index).fillna(0.0)
    return hits / docs["n_words"]


def score_tfidf(counts: pd.DataFrame, docs: pd.DataFrame,
                word_list: set[str]) -> pd.Series:
    """Equation (1). See the module docstring for the formula and the log base.

    `df_i` and `N` are taken from `docs`, so whatever sample you pass in *is* the
    corpus. Pass the final filtered sample.
    """
    N = len(docs)
    # Restrict to this corpus first: df_i must not count documents that were
    # filtered out, or the idf term is computed against a different N.
    c = counts[counts["accession"].isin(docs.index)]
    c = c[c["word"].isin(word_list)]
    if c.empty:
        return pd.Series(0.0, index=docs.index)

    df_i = c.groupby("word")["accession"].nunique()
    idf = np.log(N / df_i)                       # 0 for a word in every filing

    a_j = docs["n_words"] / docs["n_distinct"]   # average count within document

    tf_term = 1.0 + np.log(c["count"].to_numpy(dtype=float))
    norm = 1.0 + np.log(a_j.reindex(c["accession"]).to_numpy(dtype=float))
    weights = (tf_term / norm) * idf.reindex(c["word"]).to_numpy(dtype=float)

    out = pd.Series(weights, index=c["accession"].to_numpy()).groupby(level=0).sum()
    return out.reindex(docs.index).fillna(0.0)


def score_corpus(counts: pd.DataFrame, meta: pd.DataFrame,
                 word_lists: dict[str, set[str]]) -> pd.DataFrame:
    """Both measures, both weightings, for one corpus.

    Returns four columns: neg_prop, unc_prop, neg_tfidf, unc_tfidf.
    """
    docs = meta.set_index("accession")[["n_words", "n_distinct"]]
    out = pd.DataFrame(index=docs.index)
    for short, cat in (("neg", "Negative"), ("unc", "Uncertainty")):
        wl = word_lists[cat]
        out[f"{short}_prop"] = score_proportional(counts, docs, wl)
        out[f"{short}_tfidf"] = score_tfidf(counts, docs, wl)
    return out.reset_index()


# ---------------------------------------------------------------------------
# The README's self-check
# ---------------------------------------------------------------------------
SELFCHECK_DOCS = {
    "d1": "LOSS LOSS RISK GAIN",
    "d2": "LOSS GAIN GAIN",
    "d3": "RISK RISK RISK GAIN",
}
SELFCHECK_LIST = {"LOSS", "RISK"}


def selfcheck() -> pd.DataFrame:
    """Reproduce the README's worked example through the real scoring functions.

    Expected: proportional 0.7500 / 0.3333 / 0.7500, tf.idf 0.8480 / 0.2885 /
    0.5026. Getting 0.368 for d1 means log base 10 crept in somewhere.
    """
    rows, docs = [], []
    for name, text in SELFCHECK_DOCS.items():
        c = Counter(text.split())
        docs.append({"accession": name,
                     "n_words": sum(c.values()),
                     "n_distinct": len(c)})
        rows.extend((name, w, n) for w, n in c.items())

    counts = pd.DataFrame(rows, columns=["accession", "word", "count"])
    meta = pd.DataFrame(docs)
    idx = meta.set_index("accession")[["n_words", "n_distinct"]]

    return pd.DataFrame({
        "a_j": idx["n_words"] / idx["n_distinct"],
        "proportional": score_proportional(counts, idx, SELFCHECK_LIST),
        "tfidf": score_tfidf(counts, idx, SELFCHECK_LIST),
    })


if __name__ == "__main__":
    print(selfcheck().round(4))
