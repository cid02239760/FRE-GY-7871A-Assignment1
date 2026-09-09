"""Hand-checked cases for the day-0 rule.

The rule: day 0 is the first trading day on or after the later of (a) the EDGAR
filing date and (b) the acceptance date, pushed forward one day when acceptance
lands at or after 16:00 Eastern. acceptance_datetime is UTC, so it is converted
first -- that conversion is the whole point, and getting it wrong shifts a large
share of the sample by one session without any error being raised.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.events import assign_day0  # noqa: E402


# Feb/Mar 2021. 2021-02-27 and -28 are a weekend; 2021-03-06 and -07 likewise.
CAL = pd.DatetimeIndex([
    "2021-02-24", "2021-02-25", "2021-02-26",     # Wed, Thu, Fri
    "2021-03-01", "2021-03-02", "2021-03-03",     # Mon, Tue, Wed
    "2021-03-04", "2021-03-05", "2021-03-08",
])

CASES = [
    # (label, filing_date, acceptance UTC, expected day0, expected moved)
    ("after close Friday -> Monday",
     "2021-02-26", "2021-02-26T21:19:13Z", "2021-03-01", True),
    ("before close -> same session",
     "2021-02-25", "2021-02-25T14:30:00Z", "2021-02-25", False),
    ("exactly 16:00 ET counts as after close",
     "2021-02-25", "2021-02-25T21:00:00Z", "2021-02-26", True),
    ("15:59 ET is still tradable that session",
     "2021-02-25", "2021-02-25T20:59:00Z", "2021-02-25", False),
    ("filing_date later than acceptance wins",
     "2021-03-03", "2021-02-26T14:00:00Z", "2021-03-03", False),
    ("accepted on a non-session rolls to the next one",
     "2021-02-27", "2021-02-27T14:00:00Z", "2021-03-01", False),
]


@pytest.fixture(scope="module")
def result():
    meta = pd.DataFrame({
        "label": [c[0] for c in CASES],
        "filing_date": pd.to_datetime([c[1] for c in CASES]),
        "acceptance_datetime": [c[2] for c in CASES],
    })
    return assign_day0(meta, CAL).set_index("label")


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_day0(result, case):
    label, _, _, expected_day0, expected_moved = case
    row = result.loc[label]
    assert row["day0"] == pd.Timestamp(expected_day0)
    assert bool(row["day0_moved"]) is expected_moved


def test_utc_conversion_actually_happens(result):
    """21:19 UTC is 16:19 in New York, not 21:19.

    Skipping the tz conversion would read 21:19 as after close too, so this
    checks the ET hour directly rather than the outcome it happens to share.
    """
    assert result.loc["after close Friday -> Monday", "acceptance_et"].hour == 16


def test_missing_acceptance_falls_back_to_filing_date():
    meta = pd.DataFrame({
        "filing_date": pd.to_datetime(["2021-02-25"]),
        "acceptance_datetime": [None],
    })
    out = assign_day0(meta, CAL)
    assert out.loc[0, "day0"] == pd.Timestamp("2021-02-25")
    assert not out.loc[0, "day0_moved"]
