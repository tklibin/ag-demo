from cdi import data_gen
from cdi.engine import CdiEngine
from cdi.risk_model import DenialRiskModel


def test_generator_is_deterministic():
    a, b = data_gen.generate(50, seed=1), data_gen.generate(50, seed=1)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_model_learns_denial_signal_and_round_trips(tmp_path):
    claims = data_gen.generate(600, seed=3)
    model = DenialRiskModel.train(claims)
    assert model.metrics.roc_auc > 0.75

    path = tmp_path / "model.json"
    model.save(path)
    loaded = DenialRiskModel.load(path)
    engine = CdiEngine(loaded)
    r = engine.analyze(claims[0])
    assert r.denial_probability == CdiEngine(model).analyze(claims[0]).denial_probability


def test_flagged_claims_score_higher_than_clean_ones():
    claims = data_gen.generate(600, seed=5)
    engine = CdiEngine(DenialRiskModel.train(claims))
    reports = [engine.analyze(c) for c in data_gen.generate(100, seed=6)]
    high = [r.denial_probability for r in reports if any(f.severity == "HIGH" for f in r.findings)]
    clean = [r.denial_probability for r in reports if not r.findings]
    assert sum(high) / len(high) > 2 * sum(clean) / len(clean)


def test_engine_without_model_uses_rules_only():
    claim = data_gen.generate(1, seed=9)[0]
    report = CdiEngine(model=None).analyze(claim)
    assert report.denial_probability is None and report.risk_band in {"HIGH", "LOW"}
