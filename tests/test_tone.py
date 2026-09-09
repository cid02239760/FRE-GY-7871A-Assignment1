"""The README's tf.idf self-check, run through the real scoring path.

If this fails, every number in Tables 2-6 is wrong, so it runs first.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tone import selfcheck  # noqa: E402


EXPECTED = {
    #        a_j,    proportional,  tf.idf
    "d1": (4 / 3, 0.7500, 0.8480),
    "d2": (1.500, 0.3333, 0.2885),
    "d3": (2.000, 0.7500, 0.5026),
}


@pytest.fixture(scope="module")
def result():
    return selfcheck()


@pytest.mark.parametrize("doc", ["d1", "d2", "d3"])
def test_selfcheck(result, doc):
    a_j, prop, tfidf = EXPECTED[doc]
    row = result.loc[doc]
    assert row["a_j"] == pytest.approx(a_j, abs=1e-4)
    assert row["proportional"] == pytest.approx(prop, abs=1e-4)
    assert row["tfidf"] == pytest.approx(tfidf, abs=1e-4)


def test_not_log10(result):
    """0.368 for d1 is the log-base-10 answer. Equation (1) uses natural logs."""
    assert result.loc["d1", "tfidf"] != pytest.approx(0.368, abs=1e-3)
