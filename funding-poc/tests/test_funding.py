import numpy as np
import pandas as pd
import pytest

from funding import data_gen, features
from funding.models import FundingModels
from funding.policy import Policy, backtest, make_offers, naive_offers

AS_OF = pd.Timestamp("2026-08-01")


@pytest.fixture(scope="module")
def history():
    return data_gen.generate_history(per_day=25, seed=11)


@pytest.fixture(scope="module")
def models(history):
    return FundingModels.train(history, AS_OF)


@pytest.fixture(scope="module")
def test_claims(history):
    return history[history["lock_date"] >= AS_OF]


def test_generator_is_deterministic():
    a = data_gen.generate(data_gen.HISTORY_START, data_gen.HISTORY_START + pd.Timedelta(days=30), seed=1)
    b = data_gen.generate(data_gen.HISTORY_START, data_gen.HISTORY_START + pd.Timedelta(days=30), seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_history_features_use_only_events_known_before_lock(history):
    """Hiding every outcome that arrived on or after the lock date must not change the features."""
    day = pd.Timestamp("2026-03-02")
    claims = history[history["lock_date"] == day]
    hidden = history.copy()
    hidden.loc[hidden["adjudicated_date"] >= day, "adjudicated_date"] = pd.NaT
    hidden.loc[hidden["paid_date"] >= day, ["paid_date", "days_to_cash"]] = [pd.NaT, np.nan]
    pd.testing.assert_frame_equal(features.build(claims, history), features.build(claims, hidden))


def test_models_learn_and_predictions_are_sane(models, history, test_claims):
    assert models.info.approval_valid_auc > 0.75
    pred = models.predict(features.build(test_claims, history))
    assert pred["p_approve"].between(0, 1).all() and pred["paid_ratio"].between(0, 1).all()
    assert (pred["days_p90"] >= pred["days_p50"]).all()
    paid = test_claims["paid_date"].notna()
    coverage = (test_claims.loc[paid, "days_to_cash"] <= pred.loc[paid, "days_p90"]).mean()
    assert 0.8 < coverage < 0.97


def test_more_open_high_alerts_never_raise_approval(models, history, test_claims):
    X = features.build(test_claims.head(200), history)
    base = models.approval.predict(X)
    worse = models.approval.predict(X.assign(open_high_alerts=X["open_high_alerts"] + 1))
    assert (worse <= base + 1e-12).all()


def test_models_round_trip(models, history, test_claims, tmp_path):
    models.save(tmp_path)
    X = features.build(test_claims.head(100), history)
    pd.testing.assert_frame_equal(models.predict(X), FundingModels.load(tmp_path).predict(X))


def test_policy_tiers_and_pricing(models, history, test_claims):
    pred = models.predict(features.build(test_claims, history))
    offers = make_offers(test_claims, pred)
    policy = Policy()
    assert (offers.loc[offers["tier"] == "C", "advance"] == 0).all()
    assert (offers.loc[test_claims["open_high_alerts"] > 0, "tier"] == "C").all()
    assert (offers.loc[offers["tier"] == "A", "p_approve"] >= policy.tier_a_min_p).all()
    funded = offers[offers["advance"] > 0]
    # Lower approval probability must cost more: compare the riskiest and safest funded quartiles.
    q = funded["p_approve"].quantile([0.25, 0.75])
    assert (funded.loc[funded["p_approve"] <= q[0.25], "fee_rate"].mean()
            > funded.loc[funded["p_approve"] >= q[0.75], "fee_rate"].mean())


def test_model_policy_beats_naive_flat_advance(models, history, test_claims):
    offers = make_offers(test_claims, models.predict(features.build(test_claims, history)))
    assert backtest(test_claims, offers)["net"] > backtest(test_claims, naive_offers(test_claims))["net"]
