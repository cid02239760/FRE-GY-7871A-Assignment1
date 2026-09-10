# Assignment 1 Report

### Uncertainty and Sentiment Analysis of Quarterly and Annual Financial Reports

FRE-GY 7871 A · NLP and the Investment Process

**Name:** Yuri Moghaddam Nasrollahi
**NetID:** ym3414
**GitHub repo:** https://github.com/cid02239760/FRE-GY-7871A-Assignment1

---

## 1. What I Did

I measured how much negative language and how much hedging appear in the 10-K and 10-Q filings of
the SEC filers held by the six ARK ETFs between 2021 and 2025, then tested whether either measure
trends over time or predicts anything about the stock. Sentiment is the Loughran-McDonald Fin-Neg
list of 2,355 words; uncertainty is Fin-Unc, 297 words. Each is scored twice, once as a share of
the filing's total words and once with the tf.idf weighting of equation (1) in Loughran and
McDonald (2011). The final sample is 1,542 filings from 91 firms across 20 quarters, 48.8 million
words in all.

The short version of what I found is that these measures describe what kind of company is filing.
They say very little about what happens to it next. Every result that survives a within-firm
specification turns out to be descriptive or cross-sectional, and every within-firm predictive test
comes back null.

## 2. Data Construction

The universe starts from the 124 US-listed tickers in the frozen ARK holdings snapshot. Seven have
no CIK on file, being foreign private issuers or private companies, and 24 have a CIK but filed no
10-K or 10-Q in the window. That leaves 93 filers and 1,702 filings. Table 1 is the waterfall from
there.

**Table 1. Sample filters**

| Filter | Removed | Remaining |
|---|---:|---:|
| 10-K and 10-Q filings, 2021-2025, 93 ARK filers | | 1,702 |
| 1a. Amendments (10-K/A, 10-Q/A), excluded at query | 46 | 1,702 |
| 1b. Failed to parse (zero words) | 0 | 1,702 |
| 2. Below word floor (2,000 for 10-K, 1,000 for 10-Q) | 0 | 1,702 |
| 3a. Dual-class listing (same filing under two tickers) | 20 | 1,682 |
| 3b. Multiple filings per company-quarter (earliest kept) | 14 | 1,668 |
| 4a. No usable day 0 | 0 | 1,668 |
| 4b. Price on day -1 below $3 | 103 | 1,565 |
| 5. Fewer than 60 trading days of returns either side | 23 | 1,542 |

Three rows need a sentence rather than a number. The 46 amendments never appear in the downloaded
metadata at all, because `src/edgar.py` filters them out at the EDGAR query; I re-queried with
amendments included so the row would report a real count instead of a zero that only means I never
looked. The 20 dual-class removals are all Alphabet, which sits in the ARK universe under both GOOG
and GOOGL: one CIK, one set of filings, counted twice. And two filters removed nothing at all. No
filing failed to parse, and none fell below the word floor, since the shortest 10-K runs to 17,774
words and the shortest 10-Q to 3,388. Those floors are insurance against a parser failure that did
not happen, and I think reporting them as zeroes is more informative than leaving them out.

I used the starter parser unchanged, so inline-XBRL scaffolding is stripped and any table is dropped
whose non-space characters are more than 15% digits. Both are choices. The threshold splits the
difference between Loughran and McDonald's exclusion of all tables and the awkward fact that modern
filings use tables for page layout, where a blanket rule would throw away real narrative. Nothing in
the sample failed to parse. 96.3% of filings matched a point-in-time share count, and the 3.7% that
did not are left missing rather than filled forward from a later filing, so they drop out of any
regression using market capitalisation.

On point-in-time: day 0 is the first trading day on or after the later of the EDGAR filing date and
the acceptance date, with acceptance pushed forward one day when it lands at or after 16:00 Eastern.
`acceptance_datetime` arrives in UTC and has to be converted first. The size of this surprised me.
1,286 filings, 75.6% of the sample, were accepted at or after 16:00 Eastern, and the rule moves day
0 for 951 filings, 55.9%. Treating the EDGAR filing date as tradable would have put more than half
the sample on the wrong day. The gap between those two percentages is filings whose
acceptance-plus-one lands on a weekend that the naive rule was already rolling past.

Share counts come from `dei:EntityCommonStockSharesOutstanding` as printed on each filing, joined on
accession number, so market capitalisation on day -1 uses the count that was actually on the
document being scored.

## 3. Word Lists

Fin-Neg has 2,355 words and Fin-Unc 297. Forty words appear on both, 13% of the uncertainty list,
among them *doubt*, *doubtful*, *instability*, *risky*, *unpredictable*, *unexpected*, *volatile*
and *volatility*. The shared words are interesting in themselves. They are the terms where being
unsure and being in trouble amount to the same statement: a firm describing its results as
*volatile* is not making a neutral claim about a distribution. That overlap puts a floor under how
independent the two measures can possibly be, and it is one of three reasons they correlate as
highly as they do (§5).

## 4. Method

The proportional measure is the count of list words divided by the filing's total word count. The
tf.idf measure is equation (1), with natural logs throughout:

> w<sub>i,j</sub> = [ (1 + ln tf<sub>i,j</sub>) / (1 + ln a<sub>j</sub>) ] × ln( N / df<sub>i</sub> ) for tf<sub>i,j</sub> > 0, and 0 otherwise

Here a<sub>j</sub> is the average word count *within* document j, meaning total words over distinct
words; N is the number of documents in the corpus and df<sub>i</sub> the number containing word i.
The score for a filing is the sum of w<sub>i,j</sub> over the words on the list. Two terms are
ambiguous, and I read them this way: a<sub>j</sub> is within-document rather than a corpus average,
and the score is an unweighted sum rather than a mean or a length-normalised quantity. The README's
three-document self-check confirms both readings, since my implementation returns 0.8480, 0.2885
and 0.5026. That check runs in `tests/test_tone.py` against the same functions that score the real
corpus, so it cannot quietly drift.

N and df are computed on the final filtered sample, and those scores are used for every exhibit and
regression. Computing them on the full 1,702 and then regressing on 1,542 would make the idf term
refer to a population the regressions never actually see.

I never pool the two form types when describing a trend. 10-Ks are twice as long as 10-Qs, heavier
in risk language, and they cluster in Q1, so a pooled quarterly mean carries an annual sawtooth that
is a pure calendar artefact. Figure 1 plots them separately and Table 4 reports them separately,
alongside a within-firm test.

The aggregate trend test regresses the quarterly mean on a linear time index with Newey-West
standard errors, bandwidth 2 by the usual 4(T/100)^(2/9) rule at T = 20. It is needed here because
twenty observations of a persistent series regressed on time will produce a large t statistic
whether or not anything is actually happening. The within-firm test carries firm fixed effects with
errors clustered by firm, since a firm's twenty filings are one story repeated rather than twenty
independent draws. The panel tests in Tables 5 and 6 use two-way clustering by firm and quarter:
firm for the reason just given, and quarter because every filing in a given quarter shared a market,
and a common shock is exactly the thing a cross-sectional regression mistakes for a result. All
regressors of interest are standardised, so coefficients read per standard deviation.

## 5. What the Measures Are Made Of

**Table 2. Summary statistics** (proportional measures, % of words)

| Form | n | Fin-Neg mean | Fin-Neg sd | Fin-Unc mean | Fin-Unc sd |
|---|---:|---:|---:|---:|---:|
| 10-K | 384 | 2.58 | 0.54 | 2.14 | 0.36 |
| 10-Q | 1,158 | 2.18 | 1.07 | 1.95 | 0.66 |
| All | 1,542 | 2.28 | 0.98 | 2.00 | 0.60 |

**Table 3 (summary). Concentration of each list**

| List | Size | Words observed | Top 10 share | Top 30 share |
|---|---:|---:|---:|---:|
| Fin-Neg | 2,355 | 1,565 | 27.4% | 46.1% |
| Fin-Unc | 297 | 243 | 78.1% | 92.4% |

### Q1. What is each measure actually made of?

Fin-Unc is essentially two words. MAY alone is 37.3% of the entire uncertainty count and COULD
another 20.6%, so 58% between them. The top ten words carry 78.1% and the top thirty 92.4%. Only
243 of the 297 listed words appear anywhere in 48.8 million words.

Those words are not management hedging in the sense the construct intends. MAY, COULD, RISK and
RISKS each appear in 100% of filings, APPROXIMATELY in 94.9%. They are the standing furniture of a
risk-factor section that is largely copied forward from the previous filing. What the proportional
uncertainty measure mostly tracks, then, is how much of a filing is risk-factor boilerplate, not
how uncertain management sounds this particular quarter. This is exactly the case for tf.idf
weighting: a word appearing in every filing gets ln(N/df) = ln(1) = 0 and drops out completely, so
the two schemes end up measuring genuinely different things rather than the same thing with more or
less precision.

Fin-Neg is much better spread. Its top ten carry 27.4%, and 1,565 of its 2,355 words appear
somewhere. Its leading terms (LOSS, ADVERSELY, ADVERSE, CLAIMS, HARM) are still risk-factor
vocabulary, but no single word dominates the way MAY does.

### Q2. Are sentiment and uncertainty measuring different things?

Largely not. The two proportional measures correlate 0.881 pooled and the two tf.idf measures
0.933. Within form, the proportional correlation is 0.847 for 10-Ks and 0.882 for 10-Qs, so this
is not a 10-K/10-Q composition effect.

Three things drive it. There are the 40 shared words, 13% of Fin-Unc. There is shared exposure to
how much risk-factor prose a filing carries, which moves both numerators together. And there is
genuine co-movement, since filings that discuss bad news also tend to hedge.

The filings that separate the two are worth looking at. The most
uncertainty-relative-to-sentiment filings in the sample are Alphabet's 10-Qs: short documents of
around 9,500 words scoring 2.4-2.6% on Fin-Unc against only 1.3% on Fin-Neg. That is roughly what
you would expect from a highly profitable company with little bad news to report but a great deal
of forward-looking *may* and *could* language about products, litigation and regulation. It is a
useful check on what the uncertainty measure captures, which is hedging about the future,
separable from present adversity at least at the extremes.

What this means for the rest of the report is that Tables 5 and 6 are not two independent tests.
They use different dependent variables, which is what keeps them distinct at all, but their
right-hand-side variables are close to being the same variable. A finding on one should not be read
as corroborating a finding on the other.

## 6. Trends, 2021-2025

![Figure 1. Tone by quarter, 2021-2025, by form type, with the VIX](outputs/figure1.png)

*Figure 1. Both proportional measures by quarter, separately by form type, with the VIX behind
them.*

The composition corrections described in §4 are applied before anything is described: form types
separated, and the within-firm test reported alongside the aggregate one.

**Table 4. Trend tests** (proportional measures, percentage points per year)

| Measure | Sample | Test | pp/year | t (OLS) | t (NW) |
|---|---|---|---:|---:|---:|
| Fin-Neg | 10-K | aggregate | +0.056 | 1.95 | 3.40 |
| Fin-Neg | 10-Q | aggregate | -0.032 | -2.28 | -2.18 |
| Fin-Neg | pooled | within firm | -0.011 | | -0.72 |
| Fin-Unc | 10-K | aggregate | +0.027 | 1.69 | 2.35 |
| Fin-Unc | 10-Q | aggregate | -0.025 | -2.89 | -3.21 |
| Fin-Unc | pooled | within firm | -0.011 | | -1.35 |

### Q3. Did either measure trend?

No trend that survives the obvious checks. The aggregate tests point in opposite directions
depending on form type: Fin-Neg rises 0.056 pp/year in 10-Ks with a Newey-West t of 3.40 and falls
0.032 pp/year in 10-Qs with a t of -2.18, and Fin-Unc does the same thing. Two significant
coefficients of opposite sign on what is supposed to be one question should be read as a warning
rather than as two findings.

The within-firm test is the one to lead with, since it asks whether a given company's language
changed rather than which companies happened to file in which quarter. It is null for both
measures: -0.011 pp/year for Fin-Neg (t = -0.72) and -0.011 for Fin-Unc (t = -1.35). This is the
specification I trust, and it is not underpowered. With 1,542 filings and 91 firms it would have
picked up a trend of a few hundredths of a percentage point a year.

Newey-West does real work here, though not always in the direction people expect. For 10-K Fin-Neg
the OLS t is 1.95 and the Newey-West t is 3.40, because the residual autocorrelation at short lags
happens to be negative. Correcting for autocorrelation can raise a t statistic as easily as lower
it, and writing as though HAC only ever shrinks things would misrepresent what it does.

The tf.idf trends should not be read as tone at all, because they are largely an artefact of
document length. tf.idf correlates 0.95 with length, which is unsurprising given that the score is
a sum over the words present rather than a rate. Filings in this sample shrank within firm by 2.9%
a year (t = -2.72). Once log document length is controlled, the Fin-Neg tf.idf trend flips sign
entirely, from -0.75/year (t = -0.80) to +1.95/year (t = +4.05), and the Fin-Unc tf.idf decline
more than halves. I should say that I added that control only after seeing the raw tf.idf result,
and it changed my conclusion. I would rather report that sequence than present the check as though
I had planned it.

So which reading of the trend does my evidence support? Neither the story that companies are
hedging more nor the story that they are hedging less. It supports the duller conclusion: within
firm, across these twenty quarters, tone did not move detectably, and what movement shows up in the
aggregate is composition and document length.

## 7. Uncertainty, Volatility and Returns

**Table 5. Post-filing volatility on Fin-Unc proportional** (per SD; two-way clustered)

| Spec | β | se | t | R² |
|---|---:|---:|---:|---:|
| (1) no pre-vol control, quarter FE | +0.0484 | 0.0127 | 3.80 | 0.445 |
| (2) + pre-filing volatility | +0.0338 | 0.0092 | 3.66 | 0.579 |
| (3) + firm FE, no pre-vol | -0.0075 | 0.0068 | -1.09 | 0.734 |
| (4) + firm FE + pre-vol | -0.0082 | 0.0063 | -1.30 | 0.741 |

### Q4. Does uncertainty language predict volatility?

It depends entirely on which comparison you make, and that dependence is itself the result.

Between firms, without the pre-filing volatility control, a one-SD increase in Fin-Unc predicts
4.84 additional annualised volatility points (t = 3.80). Adding pre-filing volatility cuts that to
3.38 points (t = 3.66), an attenuation of about 30%. So roughly a third of the raw coefficient was
picking up the level of volatility rather than predicting it, which is what the brief warns about.
Volatile companies write hedged filings, and Fin-Unc correlates 0.16 with pre-filing volatility.

The remaining two-thirds does not survive firm fixed effects. The coefficient collapses to -0.008
(t = -1.09) without the pre-vol control and -0.008 (t = -1.30) with it. I think this is the more
informative comparison, and it is one the brief's pair of specifications does not make. The claim
that volatile companies write hedged filings is a claim about differences between firms, and a
single quarter of realised volatility is a noisy proxy for a firm's volatility regime, so the
pre-vol control only partly absorbs it. Firm fixed effects use all twenty quarters and absorb it
properly.

That leaves the surviving 3.38 points in column (2) as a between-firm fact. Ask instead whether
this company, writing a more hedged filing than it usually does, sees its volatility rise above its
own norm, and there is nothing there. The language marks which firms are risky. It does not
forecast when they will become more so.

**Table 6. Filing-period excess return [0,+3] on Fin-Neg proportional** (per SD)

| Spec | β | se | t | p | MDE (80% power) |
|---|---:|---:|---:|---:|---:|
| (1) quarter FE | -0.67 pp | 0.31 pp | -2.20 | 0.028 | 0.86 pp |
| (2) + firm FE | -1.17 pp | 0.62 pp | -1.89 | 0.058 | 1.73 pp |

The power arithmetic comes first. With quarter fixed effects the standard error is 0.31 pp, so the
smallest effect detectable at 80% power is 0.86 pp per SD. Adding firm fixed effects doubles the
standard error and pushes that threshold to 1.73 pp. The four-day excess return itself has a
standard deviation of about 13 pp, so both versions are hunting a signal worth a small fraction of
the noise in a single observation.

The two columns disagree, and the disagreement is the informative part. The sign is the predicted
one throughout: more negative language, lower excess return. But look at what happened when firm
fixed effects went in. The coefficient did not shrink toward zero, it nearly doubled. What changed
was precision, because with twenty quarters per firm the fixed effects absorb much of the
identifying variation and the whole cost lands in the standard error. This is a null result in the
demanding specification, and a null here is a correct answer, but it is a null of a particular
kind. The finding is not that there is no relationship. It is that there is an effect of roughly
the size theory predicts, measured with a standard error of the same magnitude.

## 8. 10-K versus 10-Q

### Q5. Do 10-Qs behave like 10-Ks?

Start with the variance of a proportional measure. The intuition that 10-Qs are shorter and more
templated predicts less dispersion. The data say the opposite. Fin-Neg proportional has an sd of
0.54 pp in 10-Ks and 1.07 pp in 10-Qs, a coefficient of variation of 0.21 against 0.49. The reason
is that a 10-Q's risk-factor section is optional. Many carry a brief "no material changes"
cross-reference to the 10-K while others reproduce the section in full, so the 10-Q population is
bimodal. That is dispersion in composition rather than in tone, and it is a reason to distrust the
10-Q measure rather than to prefer it for having more variance.

Is the filing-date reaction separately identified for a 10-Q? Largely not. A 10-Q reaches day 0 a
median of 36 days after the period it covers, against 54 days for a 10-K, and the earnings release
for that quarter almost always precedes it by several days. The [0,+3] window therefore opens after
the market has already priced the quarter's news, so the test becomes whether the document carries
information incremental to the earnings call. That is a considerably weaker hypothesis than the
design appears to be testing. The same problem exists for 10-Ks but is milder.

Which form should carry more textual signal? The 10-K, and the volatility results are consistent
with that. Run separately, neither form gives a significant coefficient once firm fixed effects are
in, but the 10-K is where the measure is better behaved: lower dispersion, a mandatory and complete
risk-factor section, and a filing date that is not immediately preceded by a fuller disclosure.

## 9. Limitations

Survivorship is the big one, and it has a number attached. The universe is a frozen present-day
snapshot of ARK holdings, which means it is the set of companies that survived to 2026 and were
still worth holding. Of 124 tickers, 93 are SEC filers with history in the window and 91 reach the
final sample. Nothing that delisted, was acquired, or collapsed between 2021 and 2025 appears at
all.

For the trend work this bites harder than the usual caveat suggests. Survivors are the firms that
did not blow up. If deteriorating firms write more negative and more hedged filings before they
fail, then excluding all of them removes precisely the observations that would have produced an
upward trend. A null trend in a survivor-only panel is close to the expected result, and I would
not read the absence of a trend as evidence about the population of US filers generally.

Twenty quarters is also a short series. The aggregate test has 20 observations of a persistent
series, and Newey-West helps with the standard error but not with the fact that one unusual year
can dominate the fitted slope.

On the benchmark, SPY is standing in for the market. These are high-beta growth names, so a
market-model or characteristic-matched benchmark would leave less systematic variation in the
residual and would probably tighten Table 6.

For completeness, here is every specification I ran, not only the reported ones: proportional and
tf.idf for both lists; volatility with and without the pre-filing control, each with and without
firm fixed effects; trends aggregate and within firm, pooled and by form; tf.idf trends with and
without a length control; volatility and returns split by form type; and returns with and without
firm fixed effects. The length control, as noted above, was added after seeing the raw tf.idf
trend.

### Q6. Which of my results do I believe?

This is not the same question as which of them are significant, so I will go test by test.

The concentration of Fin-Unc I believe. It is a descriptive count over 48.8 million words with no
inference in it at all. MAY and COULD really are 58% of the measure.

The high correlation between the two measures I also believe. It is descriptive, it is stable
across form types, and the 40 shared words underwrite a good part of it mechanically.

The aggregate trends I do not believe. They have opposite signs by form type, they rest on 20
observations, and the tf.idf versions are substantially a length artefact. This is the test the
brief warns is the easiest place to find a result that is not there, and it behaved exactly that
way.

The within-firm null trend I believe as a null, with the survivorship caveat attached. It is well
powered. But it is a null about survivors, and survivors are not the population the question
implies.

The between-firm volatility result is one where I believe the fact but not the obvious reading of
it. It is strongly significant and it survives its control, yet firm fixed effects show it to be a
cross-sectional marker. I believe that hedged writers are volatile companies. I do not believe that
hedged language predicts a rise in volatility.

The within-firm volatility null I believe. It is well powered, and the coefficient sits near zero
rather than merely failing to clear a threshold.

The return test I do not believe in either direction. The sign is right in both columns, the
magnitude is comparable to its own detection threshold, and it is significant with quarter fixed
effects but not with firm fixed effects. Since the coefficient grew rather than shrank across that
change, the lost significance is lost precision and not evidence against the effect. I would not
report column (1) as a finding, and I would not report column (2) as a refutation.

Taken together this is a coherent pattern rather than a pile of failures. Everything that survives
is descriptive or cross-sectional, and every within-firm predictive test is null.

## 10. What I Would Do Next

The change with the most leverage would be rebuilding the universe from ARK's *historical* holdings
files, quarter by quarter, instead of a present-day snapshot, so that firms enter and leave the
sample as they actually did. That removes the survivorship problem undermining every trend
statement in this report. It would cost a few hours of scraping ARK's daily holdings archive, plus
some care in mapping tickers to CIKs at the right point in time.

The cheaper second option is to score the change in a filing against the same firm's previous
filing rather than its level. Most of what Table 3 exposes is boilerplate carried forward unchanged,
and differencing against the prior filing would measure new hedging instead of accumulated
template.
