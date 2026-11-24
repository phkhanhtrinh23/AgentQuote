import math

import numpy as np
import pandas as pd
import pytest

from agentquote.adapters import get_adapter
from agentquote.choice import ltt, n_min_floor, pvalue_binom, pvalue_hyper
from agentquote.conformal import ConformalQuoter, conformal_quantile
from agentquote.config import BACKBONE_NAMES, Settings
from agentquote.pipeline import AgentQuote
from agentquote.testbed import BigQueryTestbed, EarthEngineTestbed
from agentquote.types import Session


@pytest.fixture(scope="module")
def ee():
    return EarthEngineTestbed(seed=1, scale=0.15).generate()


def test_session_roundtrip(ee):
    _, sess = ee
    s = sess[0]
    t = Session.from_dict(s.to_dict())
    assert np.allclose(s.S(), t.S())
    parts = s.parts()
    assert np.allclose(sum(parts.values()), s.S())


def test_conformal_joint_coverage():
    rng = np.random.default_rng(0)
    n, cov = 200, []
    for _ in range(300):
        Sh = np.exp(rng.normal(0, 1, (n + 1, 3)))
        S = Sh * np.exp(rng.normal(0, 0.5, (n + 1, 3)))
        q = ConformalQuoter("joint", 0.1).fit(S[:n], Sh[:n])
        cov.append(q.covered(S[n:], Sh[n:])[0])
    assert 0.86 < np.mean(cov) < 0.95


def test_conformal_quantile_rank():
    assert conformal_quantile(np.arange(1, 10), 0.1) == 9
    assert math.isinf(conformal_quantile(np.arange(1, 5), 0.1))


def test_ltt_pvalues():
    assert pvalue_binom(0, 100, 0.2) < 1e-9
    assert pvalue_hyper(5, 100, 300, 0.2) <= 1
    L = np.zeros((200, 5), bool)
    L[:10, 3:] = True
    L[:80, 4] = True
    h, _ = ltt(L, 0.2, 0.1)
    assert h == 3
    assert n_min_floor(0.2, 0.1) > 5


def test_world_model_and_pipeline(ee):
    _, sess = ee
    units = sorted({s.unit for s in sess})
    fit = [s for s in sess if s.unit in set(units[::2])]
    rest = [s for s in sess if s.unit not in set(units[::2])]
    pq = AgentQuote(get_adapter("earthengine"), Settings(n_roll=100)).fit(fit)
    df = pq.estimate(rest, np.random.default_rng(0))
    assert np.isfinite(df[["Shat_comp", "Shat_tok", "Shat_time"]].values).all()
    half = df["unit"].isin(sorted(df["unit"].unique())[::2])
    cal = pq.calibrate(df[half], df[~half & df["has_ref"]])
    quotes, pick = pq.quote_and_choose(df[~half], cal)
    assert len(quotes) == (~half).sum()
    assert set(pick["request_id"]) == set(df.loc[~half, "request_id"])


def test_bigquery_testbed():
    _, sess = BigQueryTestbed(seed=0, scale=0.2).generate()
    ad = get_adapter("bigquery")
    for s in sess[:20]:
        for a in s.attempts:
            if a.outcome in ("validation", "runtime"):
                assert a.step.calls[0].comp == 0.0
    assert {s.b for s in sess} == set(BACKBONE_NAMES)
    v, z = ad.quantities(sess[0].plan)
    assert ad.price(v, z) > 0
