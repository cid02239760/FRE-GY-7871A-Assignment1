# Assignment 1 Report

### Uncertainty and Sentiment Analysis of Quarterly and Annual Financial Reports

FRE-GY 7871 A · NLP and the Investment Process

**Name:** Yuri Moghaddam Nasrollahi
**NetID:** ym3414
**GitHub repo:** https://github.com/cid02239760/FRE-GY-7871A-Assignment1

---

## 1. What I Did

I measured how much negative language and how much hedging appears in the 10-K and 10-Q filings of
the SEC filers held by the six ARK ETFs, over 2021–2025, and tested whether either measure trends
over time or predicts anything about the stock. Sentiment is the Loughran–McDonald Fin-Neg list
(2,355 words), uncertainty is Fin-Unc (297 words), and each is scored two ways: as a share of the
filing's total words, and with the tf.idf weighting of equation (1) in Loughran and McDonald
(2011). The final sample is 1,542 filings from 91 firms across 20 quarters, 48.8 million words in
total. The headline finding is that the measures describe *what kind of company* is filing rather
than *what is about to happen to it*: everything that survives a within-firm specification is
descriptive or cross-sectional, and every within-firm predictive test is null.

## 2. Data Construction

The universe starts from the 124 US-listed tickers in the frozen ARK holdings snapshot. Seven have
no CIK on file (foreign private issuers or private companies) and 24 have a CIK but no 10-K or 10-Q
in the window, leaving 93 filers and 1,702 filings. Table 1 is the waterfall from there.

**Table 1. Sample filters**

| Filter | Removed | Remaining |
|---|---:|---:|
| 10-K and 10-Q filings, 2021–2025, 93 ARK filers | — | 1,702 |
| 1a. Amendments (10-K/A, 10-Q/A), excluded at query | 46 | 1,702 |
| 1b. Failed to parse (zero words) | 0 | 1,702 |
| 2. Below word floor (2,000 for 10-K, 1,000 for 10-Q) | 0 | 1,702 |
| 3a. Dual-class listing (same filing under two tickers) | 20 | 1,682 |
| 3b. Multiple filings per company-quarter (earliest kept) | 14 | 1,668 |
| 4a. No usable day 0 | 0 | 1,668 |
| 4b. Price on day −1 below $3 | 103 | 1,565 |
| 5. Fewer than 60 trading days of returns either side | 23 | **1,542** |

Three of these rows deserve a sentence rather than a number. The 46 amendments never appear in the
downloaded metadata at all, because `src/edgar.py` filters them at the EDGAR query; I re-queried
with amendments included so that the row reports a real count rather than a zero that only means I
never looked. The 20 dual-class removals are Alphabet, which sits in the ARK universe under both
GOOG and GOOGL — one CIK, one set of filings, counted twice. And two filters removed nothing: no
filing failed to parse, and none fell below the word floor, since the shortest 10-K runs to 17,774
words and the shortest 10-Q to 3,388. Those floors are insurance against a parser failure that did
not occur, and reporting them as zeroes is more informative than omitting them.

**Parsing.** I used the starter parser unchanged: inline-XBRL scaffolding stripped, and any table
dropped whose non-space characters are more than 15% digits. Both are choices. The threshold is a
compromise between Loughran and McDonald's exclusion of all tables and the fact that modern filings
use tables for page layout, so a blanket rule would discard real narrative. Nothing in the sample
failed to parse, and 96.3% of filings matched a point-in-time share count; the 3.7% that did not
are left missing rather than filled forward from a later filing, and they drop out of any
regression using market capitalisation.

**Point-in-time.** Day 0 is the first trading day on or after the later of the EDGAR filing date
and the acceptance date, with acceptance pushed forward one day when it lands at or after 16:00
Eastern. `acceptance_datetime` arrives in UTC and is converted first. This matters more than it
sounds: **1,286 filings (75.6%) were accepted at or after 16:00 Eastern, and the rule moves day 0
for 951 filings, 55.9% of the sample.** Treating the EDGAR filing date as tradable would have put
more than half the sample on the wrong day. The gap between 75.6% and 55.9% is filings whose
acceptance-plus-one lands on a weekend that the naive rule was already rolling past.

Share counts come from `dei:EntityCommonStockSharesOutstanding` as printed on each filing, joined
on accession number, so market capitalisation on day −1 uses the count that was actually on the
document being scored.

## 3. Word Lists

Fin-Neg has 2,355 words and Fin-Unc 297. **Forty words appear on both** — 13% of the uncertainty
list — among them *doubt*, *doubtful*, *instability*, *risky*, *unpredictable*, *unexpected*,
*volatile* and *volatility*. The shared words are revealing in themselves: they are the terms where
being unsure and being in trouble are the same statement. A firm describing its results as
*volatile* is not making a neutral distributional claim. That overlap puts a floor under how
independent the two measures can be, and it is one of three reasons they correlate as highly as
they do (§5).

## 4. Method

The proportional measure is the count of list words divided by the filing's total word count. The
tf.idf measure is equation (1), with natural logs throughout:

> w<sub>i,j</sub> = [ (1 + ln tf<sub>i,j</sub>) / (1 + ln a<sub>j</sub>) ] × ln( N / df<sub>i</sub> ) for tf<sub>i,j</sub> > 0, and 0 otherwise

where a<sub>j</sub> is the average word count *within* document j — total words over distinct words
— N is the number of documents in the corpus and df<sub>i</sub> the number containing word i. The
score for a filing is the sum of w<sub>i,j</sub> over the words on the list. Two terms are
ambiguous and I read them as follows: a<sub>j</sub> is within-document (not a corpus average), and
the score is an unweighted **sum** rather than a mean or a length-normalised quantity. The
README's three-document self-check confirms both readings — my implementation returns 0.8480,
0.2885 and 0.5026 — and that check runs in `tests/test_tone.py` against the same functions that
score the real corpus, so it cannot drift.

N and df are computed on the **final filtered sample**, and those scores are used for every exhibit
and regression. Computing them on the full 1,702 and regressing on 1,542 would make the idf term
refer to a population the regressions never see.

**Aggregation and composition.** I never pool the two form types when describing a trend. 10-Ks are
twice as long as 10-Qs, heavier in risk language, and cluster in Q1, so a pooled quarterly mean
carries an annual sawtooth that is a pure calendar artefact. Figure 1 plots them separately, and
Table 4 reports each separately alongside a within-firm test.

**Specifications and standard errors.** The aggregate trend test regresses the quarterly mean on a
linear time index with **Newey–West** standard errors, bandwidth 2 by the usual 4(T/100)^(2/9) rule
at T = 20. It is required here because twenty observations of a persistent series regressed on time
will produce a large t statistic whether or not anything is happening. The within-firm test carries
firm fixed effects with errors clustered by firm, because a firm's twenty filings are one story
repeated, not twenty independent draws. The panel tests in Tables 5 and 6 use **two-way clustering
by firm and quarter** — firm for the reason just given, and quarter because every filing in a given
quarter shared a market, and a common shock is exactly what a cross-sectional regression mistakes
for a result. All regressors of interest are standardised, so coefficients read per standard
deviation.

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
| Fin-Unc | 297 | 243 | **78.1%** | 92.4% |

### Q1. What is each measure actually made of?

Fin-Unc is essentially two words. **MAY alone is 37.3% of the entire uncertainty count and COULD
another 20.6%** — 58% between them. The top ten words carry 78.1%, the top thirty 92.4%, and only
243 of the 297 listed words appear anywhere in 48.8 million words.

Those words are not management hedging in the sense the construct intends. MAY, COULD, RISK and
RISKS each appear in 100% of filings and APPROXIMATELY in 94.9%. They are the standing furniture of
a risk-factor section that is largely copied forward from the previous filing. What the
proportional uncertainty measure mostly tracks is *how much of a filing is risk-factor
boilerplate*, not how uncertain management sounds this quarter. This is precisely the case for
tf.idf weighting: a word in every filing gets ln(N/df) = ln(1) = 0 and drops out entirely, so the
two schemes are measuring genuinely different things rather than the same thing more or less
precisely.

Fin-Neg is far better spread — top ten 27.4%, and 1,565 of 2,355 words observed. Its leading terms
(LOSS, ADVERSELY, ADVERSE, CLAIMS, HARM) are still risk-factor vocabulary, but no single word
dominates the way MAY does.

### Q2. Are sentiment and uncertainty measuring different things?

Largely not. The two **proportional** measures correlate **0.881** pooled; the two **tf.idf**
measures correlate **0.933**. Within form the proportional correlation is 0.847 for 10-Ks and 0.882
for 10-Qs, so this is not a 10-K/10-Q composition effect.

Three things drive it: the 40 shared words (13% of Fin-Unc); shared exposure to how much
risk-factor prose a filing carries, which moves both numerators together; and genuine co-movement,
since filings discussing bad news also hedge.

The filings that separate the two are informative. The most uncertainty-relative-to-sentiment
filings in the sample are **Alphabet's 10-Qs** — short documents (~9,500 words) scoring 2.4-2.6% on
Fin-Unc against only 1.3% on Fin-Neg. That is the pattern one should expect from a highly
profitable company with little bad news to report but a great deal of forward-looking *may* and
*could* language about products, litigation and regulation. It is a useful check on what the
uncertainty measure captures: hedging about the future, genuinely separable from adversity in the
present, at least at the extremes.

The consequence for the rest of this report, stated plainly: Tables 5 and 6 are not two independent
tests. They use different dependent variables, which is what keeps them distinct, but their
right-hand-side variables are close to the same variable. A finding on one should not be read as
corroborating a finding on the other.

## 6. Trends, 2021–2025

![Figure 1. Tone by quarter, 2021-2025, by form type, with the VIX](outputs/figure1.png)

*Figure 1. Both proportional measures by quarter, separately by form type, with the VIX behind
them.*

Figure 1 plots both proportional measures by quarter, separately by form type, with the VIX behind
them. The composition corrections described in §4 are applied before anything is described:
form types separated, and the within-firm test reported alongside the aggregate one.

**Table 4. Trend tests** (proportional measures, percentage points per year)

| Measure | Sample | Test | pp/year | t (OLS) | t (NW) |
|---|---|---|---:|---:|---:|
| Fin-Neg | 10-K | aggregate | +0.056 | 1.95 | **3.40** |
| Fin-Neg | 10-Q | aggregate | −0.032 | −2.28 | **−2.18** |
| Fin-Neg | pooled | within firm | −0.011 | — | −0.72 |
| Fin-Unc | 10-K | aggregate | +0.027 | 1.69 | **2.35** |
| Fin-Unc | 10-Q | aggregate | −0.025 | −2.89 | **−3.21** |
| Fin-Unc | pooled | within firm | −0.011 | — | −1.35 |

### Q3. Did either measure trend?

**No trend that survives the obvious checks.** The aggregate tests point in *opposite directions*
by form type: Fin-Neg rises 0.056 pp/year in 10-Ks (NW t = 3.40) and falls 0.032 pp/year in 10-Qs
(t = −2.18), and Fin-Unc does the same. Two significant coefficients of opposite sign on the same
underlying question is a warning, not two findings.

The within-firm test — the one to lead with, because it asks whether a given company's language
changed rather than which companies happened to file — is **null for both measures**: −0.011
pp/year for Fin-Neg (t = −0.72) and −0.011 for Fin-Unc (t = −1.35). This is the specification I
trust, and it is not underpowered: with 1,542 filings and 91 firms it would have detected a trend
of a few hundredths of a percentage point a year.

Newey–West does real work here but not always downward. For 10-K Fin-Neg the OLS t is 1.95 and the
Newey–West t is 3.40, because the residual autocorrelation at short lags is negative. HAC is a
correction, not a haircut, and reporting it as though it only ever shrinks t statistics would
misrepresent what it does.

**The tf.idf trends are largely a length artefact, and should not be read as tone.** tf.idf
correlates **0.95** with document length — unsurprising, since the score is a sum over words
present rather than a rate. Filings shrank within firm by 2.9% a year (t = −2.72). Once log
document length is controlled, the Fin-Neg tf.idf trend **flips sign**, from −0.75/year (t = −0.80)
to +1.95/year (t = +4.05), and the Fin-Unc tf.idf decline more than halves. I added that control
after seeing the raw tf.idf result, and it changed the conclusion; that sequence is reported here
rather than presented as though the check had been planned.

**Which reading of the trend does my evidence support?** Neither the "companies are hedging more"
story nor its opposite. It supports the duller conclusion: within firm, over these twenty quarters,
tone did not move detectably, and the aggregate movements are composition and document length.

## 7. Uncertainty, Volatility and Returns

**Table 5. Post-filing volatility on Fin-Unc proportional** (per SD; two-way clustered)

| Spec | β | se | t | R² |
|---|---:|---:|---:|---:|
| (1) no pre-vol control, quarter FE | +0.0484 | 0.0127 | **3.80** | 0.445 |
| (2) + pre-filing volatility | +0.0338 | 0.0092 | **3.66** | 0.579 |
| (3) + firm FE, no pre-vol | −0.0075 | 0.0068 | −1.09 | 0.734 |
| (4) + firm FE + pre-vol | −0.0082 | 0.0063 | −1.30 | 0.741 |

### Q4. Does uncertainty language predict volatility?

The answer depends on which comparison is being made, and that *is* the answer.

**Between firms.** Without the pre-filing volatility control, a one-SD increase in Fin-Unc predicts
+4.84 annualised volatility points (t = 3.80). Adding pre-filing volatility cuts this to +3.38
points (t = 3.66), an attenuation of about 30%. So roughly a third of the raw coefficient was the
*level* of volatility rather than a prediction of it, exactly as the brief warns — volatile
companies write hedged filings, and Fin-Unc correlates 0.16 with pre-filing volatility.

**Within firm.** The remaining two-thirds does not survive firm fixed effects. The coefficient
collapses to −0.008 (t = −1.09) without the pre-vol control and −0.008 (t = −1.30) with it. This is
the more informative comparison and it is the one the brief's pair does not make: "volatile
companies write hedged filings" is a statement about differences *between* firms, and a single
quarter of realised volatility is a noisy proxy for a firm's volatility regime, so the pre-vol
control only partly absorbs it. Firm fixed effects use all twenty quarters and absorb it fully.

So the surviving 3.38 points in column (2) is a between-firm fact. Once you ask whether *this*
company writing a more hedged filing than it usually does predicts *its* volatility rising above
its own norm, there is nothing there. The language is a **marker of which firms are risky, not a
forecast of when they will become more so.**

**Table 6. Filing-period excess return [0,+3] on Fin-Neg proportional** (per SD)

| Spec | β | se | t | p | MDE (80% power) |
|---|---:|---:|---:|---:|---:|
| (1) quarter FE | −0.67 pp | 0.31 pp | −2.20 | 0.028 | 0.86 pp |
| (2) + firm FE | −1.17 pp | 0.62 pp | −1.89 | 0.058 | 1.73 pp |

**The power arithmetic, stated before the interpretation.** With quarter fixed effects the standard
error is 0.31 pp, so the smallest effect detectable at 80% power is 0.86 pp per SD. Adding firm
fixed effects doubles the standard error and pushes that threshold to 1.73 pp. The four-day excess
return itself has a standard deviation of about 13 pp, so both versions are hunting a signal worth
a small fraction of the noise in a single observation.

The two columns disagree, and the disagreement is the informative part. The sign is the predicted
one throughout — more negative language, lower excess return. But note what did *not* happen when
firm fixed effects were added: the coefficient did not shrink toward zero, it nearly doubled. What
changed was precision, because with twenty quarters per firm the fixed effects absorb much of the
identifying variation and the cost is paid entirely in the standard error. This is a null result in
the demanding specification, and a null here is a correct answer — but it is a null of a specific
kind. It is not "no relationship"; it is "an effect of about the size theory predicts, measured
with a standard error of the same magnitude."

## 8. 10-K versus 10-Q

### Q5. Do 10-Qs behave like 10-Ks?

**Variance of a proportional measure.** The intuition that 10-Qs are shorter and more templated
predicts *less* dispersion. The data say the opposite: Fin-Neg proportional has an sd of 0.54 pp in
10-Ks and 1.07 pp in 10-Qs, a coefficient of variation of 0.21 against 0.49. The reason is that a
10-Q's risk-factor section is optional — many carry a brief "no material changes" cross-reference
to the 10-K while others reproduce the section in full — so the 10-Q population is bimodal. That is
dispersion in *composition*, not in tone, and it is a reason to distrust the 10-Q measure rather
than to prefer it for having more variance.

**Is the filing-date reaction separately identified for a 10-Q?** Largely not. A 10-Q reaches day 0
a median of **36 days** after the period it covers (a 10-K, 54 days), and the earnings release for that quarter
almost always precedes it by several days. The [0,+3] window therefore opens after the market has
priced the quarter's news, so the test becomes whether the *document* carries information
incremental to the earnings call — a considerably weaker hypothesis than the design appears to
test. The same problem exists for 10-Ks but is milder.

**Which form should carry more textual signal?** The 10-K, and the volatility results are
consistent with that: run separately, neither form gives a significant coefficient once firm fixed
effects are in, but the 10-K is where the measure is better behaved — lower dispersion, a mandatory
and complete risk-factor section, and a filing date not immediately preceded by a fuller
disclosure.

## 9. Limitations

**Survivorship, with a number.** The universe is a *frozen present-day snapshot* of ARK holdings:
the companies that survived to 2026 and were still worth holding. Of 124 tickers, 93 are SEC filers
with history in the window and 91 reach the final sample. Nothing that delisted, was acquired, or
collapsed between 2021 and 2025 appears at all.

For the trend work this is sharper than the usual caveat. Survivors are the firms that did not blow
up. If deteriorating firms write more negative and more hedged filings before they fail, then
excluding all of them removes precisely the observations that would have generated an upward trend.
A null trend in a survivor-only panel is close to the expected result, and I would not read the
absence of a trend as evidence about the population of US filers.

**Twenty quarters is a short series.** The aggregate test has 20 observations of a persistent
series; Newey–West helps the standard error but not the fact that one unusual year can dominate the
fitted slope.

**Benchmark choice.** SPY stands in for the market. These are high-beta growth names, so a
market-model or characteristic-matched benchmark would leave less systematic variation in the
residual and would probably tighten Table 6.

**Every specification I ran**, not only the reported ones: proportional and tf.idf for both lists;
volatility with and without the pre-filing control, each with and without firm fixed effects;
trends aggregate and within firm, pooled and by form; tf.idf trends with and without a length
control; volatility and returns split by form type; and returns with and without firm fixed
effects. The length control was added after seeing the raw tf.idf trend.

### Q6. Which of my results do I believe?

Not the same question as which are significant, so test by test.

**The concentration of Fin-Unc — believe it.** A descriptive count over 48.8 million words with no
inference in it. MAY and COULD really are 58% of the measure.

**The high correlation between the measures — believe it.** Descriptive, stable across form types,
and mechanically underwritten by the 40 shared words.

**The aggregate trends — do not believe them.** Opposite signs by form type, 20 observations, and
the tf.idf versions substantially a length artefact. This is the test the brief warns is the
easiest place to find a result that is not there, and it behaved exactly that way.

**The within-firm null trend — believe it as a null, with the survivorship caveat.** Well powered;
but a null about survivors, which is not the population the question implies.

**The between-firm volatility result — believe the fact, not the framing.** Strongly significant
and it survives its control, but firm fixed effects show it is a cross-sectional marker. I believe
"hedged writers are volatile companies." I do not believe "hedged language predicts a rise in
volatility."

**The within-firm volatility null — believe it.** Well powered, and the coefficient is near zero
rather than merely insignificant.

**The return test — believe nothing either way.** Predicted sign in both columns, magnitude
comparable to its own detection threshold, significant with quarter fixed effects and not with firm
fixed effects — and the coefficient *grew* across that change, so the lost significance is lost
precision rather than evidence against the effect. I would not report column (1) as a finding or
column (2) as a refutation.

The pattern is coherent rather than a collection of failures: everything that survives is
descriptive or cross-sectional, and every within-firm predictive test is null. The language of a
filing tells you what kind of company you are looking at; it does not tell you what is about to
happen to it.

## 10. What I Would Do Next

The single change with the most leverage is rebuilding the universe from ARK's *historical* holdings
files quarter by quarter rather than from a present-day snapshot, so firms enter and leave as they
actually did. That removes the survivorship problem undermining every trend statement, at the cost
of a few hours scraping ARK's daily holdings archive and mapping tickers to CIKs at the right point
in time.

Second, and cheaper: score the *change* in a filing against the same firm's previous filing rather
than its level. Most of what Table 3 exposes is boilerplate carried forward unchanged, and
differencing against the prior filing would measure new hedging instead of accumulated template.
