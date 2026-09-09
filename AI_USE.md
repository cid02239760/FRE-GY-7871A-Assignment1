# AI use disclosure

> **Draft — verify and edit before submitting.** This file was written by the model it describes.
> Read it against your own recollection of the session and correct anything that does not match.
> The "what I wrote myself" section in particular is yours to fill in honestly.

---

**Tools used:**

Claude (Opus 5, via Claude Code), used throughout the assignment.

**What I used them for:**

Substantially all of the analysis layer. The starter repository supplies the data pipeline
(`scripts/00`–`03` and the `src` modules that download and parse filings); everything downstream of
that was written with Claude:

- `src/tone.py` — the proportional and tf.idf scoring, including the reading of equation (1)
- `src/events.py` — the trading calendar, the day-0 rule, the event windows, the sample filters
- `src/regress.py` — the trend, volatility and return specifications and the standard-error choices
- `src/exhibits.py` — Tables 2, 3 and Figure 1
- `tests/test_tone.py`, `tests/test_events.py`
- `analysis.ipynb` and a first draft of this report, including the answers to Q1–Q6

**What I wrote myself:**

*(Fill this in. If the honest answer is "the review, the corrections listed below, and the final
edit of the prose", say that — it is a defensible answer and an accurate one. Do not overstate it.)*

**Anything the model got wrong that I had to correct:**

Six things, in the order they surfaced. Three were bugs that produced visibly wrong output; three
were errors of judgment or fabrication that would have survived into the submission unnoticed.

1. **Filter 5 rejected the entire sample.** The model implemented "at least 60 trading days of
   returns before day 0 and 60 after" by counting the observations inside the pre-filing volatility
   window `[-60, -6]`. That window is 55 returns, not 60, so the filter removed all 1,565 remaining
   filings and the final sample came out empty. The brief's 60/60 is a data-availability
   requirement about the stock's return history, separate from the volatility window length.

2. **Table 5 was specified so that it could not show what the brief asks it to show.** The first
   version put firm fixed effects in *both* columns. But "volatile companies write hedged filings"
   is a statement about differences *between* firms, and firm fixed effects remove exactly that
   variation — so the "without the control" column was already controlled, and the coefficient came
   out insignificant (t = −1.09) where the brief says to expect a large significant one. The table
   was restructured as a four-rung ladder so both the pre-vol gap and the firm-FE gap are visible.
   This turned out to be the most informative result in the report.

3. **The tf.idf trend was nearly reported as a finding about tone.** The within-firm tf.idf
   uncertainty trend is strongly significant (t = −4.55). tf.idf is an unnormalised *sum*, so it
   correlates 0.95 with document length, and filings in this sample got shorter by 2.9% a year.
   Controlling for length, the Fin-Neg tf.idf trend flips sign entirely (−0.75, t = −0.80 →
   +1.95, t = +4.05). The length control was added only after the raw result was already on the
   page.

4. **Figure 1 had three out-of-sample quarters appended to the right-hand edge.** The VIX price
   history runs from 2020Q3 to 2026Q1 while the filing sample is 2021Q1–2025Q4; plotting two
   categorical series on one axis silently appended the unmatched quarters rather than raising an
   error.

5. **Table 6 was described using only one of its two columns.** The draft reported the firm-FE
   result (t = −1.89, p = 0.058) as the finding and did not mention that the quarter-FE column is
   significant (t = −2.20, p = 0.028). Reporting only the insignificant column would have been as
   selective as reporting only the significant one.

6. **Two claims in the report draft were fabricated.** The model wrote that the Fin-Neg/Fin-Unc
   overlap includes *exposure* and *risk* (it includes neither — the actual overlap is words like
   *risky*, *doubt*, *volatility*), and that the highest uncertainty-relative-to-sentiment filings
   were "long-risk-section biotech and pre-revenue names". Checking against the data, they are
   Alphabet's 10-Qs — short, low on bad news, heavy on forward-looking *may* and *could*. Both
   claims were plausible-sounding and both were wrong, and neither would have been caught by any
   test in the repository.

The general lesson from 2, 3 and 6: the bugs announced themselves, and the judgment errors did not.
The specification error in Table 5 produced a perfectly reasonable-looking regression table that
was answering the wrong question, and the fabricated examples in §3 and §5 read exactly like the
verified ones. Anything in a generated write-up that names a specific number, word or company is
worth checking against the data before it goes in.
