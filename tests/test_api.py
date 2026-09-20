"""Smoke tests for the serving API.

Uses FastAPI's TestClient to exercise the actual endpoints without
needing a running uvicorn server. Requires that the model artifacts
have been built first (run train_model.py and copy to serving/models/).
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVING_DIR = Path(__file__).resolve().parent.parent / "serving"
MODELS_DIR = SERVING_DIR / "models"

# Skip the whole module if artifacts aren't built yet — tests should be
# runnable in CI even before someone has trained the model
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "blitz_model.joblib").exists(),
    reason="Model artifacts not found in serving/models/ — run train_model.py first",
)

sys.path.insert(0, str(SERVING_DIR))
from app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # `with TestClient(app)` triggers the FastAPI startup event,
    # which is where we load the joblib artifacts
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_play():
    """3rd-and-8, shotgun, 6-man box — a realistic mid-game pass situation."""
    return {
        "possession_team": "KC",
        "defensive_team": "BUF",
        "offense_formation": "SHOTGUN",
        "personnel_o": "1 RB, 1 TE, 3 WR",
        "personnel_d": "4 DL, 2 LB, 5 DB",
        "quarter": 3,
        "down": 3,
        "yards_to_go": 8,
        "yards_to_goal": 45,
        "defenders_in_box": 6,
        "pre_snap_home_score": 14,
        "pre_snap_visitor_score": 17,
        "game_clock": "5:32",
        "is_home_offense": 0,
    }


def test_health_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"] == "LogisticRegression"


def test_model_info_exposes_readable_coefficients(client):
    """The point of shipping a linear model: you can read it. defendersInBox should
    be the strongest numeric driver and should push the odds of a blitz UP."""
    body = client.get("/model-info").json()
    coefs = body["coefficients"]["numeric_per_standard_deviation"]
    assert "defendersInBox" in coefs
    assert coefs["defendersInBox"]["odds_ratio"] > 1.0
    # Numeric coefficients are ordered by absolute effect, strongest first.
    assert next(iter(coefs)) == "defendersInBox"


def test_model_info_keeps_the_random_forest_challenger(client):
    """The 'a tie goes to the simpler model' claim has to stay checkable."""
    body = client.get("/model-info").json()
    assert "random_forest_challenger" in body["baseline_comparison"]
    assert "logistic_regression_shipped" in body["baseline_comparison"]


def test_unseen_category_falls_back_to_zero_block(client, sample_play):
    """Unseen labels encode to -1, and OneHotEncoder(handle_unknown='ignore') turns
    that into an all-zero block rather than an arbitrary tree split."""
    weird = {**sample_play, "defensive_team": "ZZZ", "personnel_d": "9 DL, 9 LB, 9 DB"}
    r = client.post("/predict", json=weird)
    assert r.status_code == 200
    assert 0.0 <= r.json()["blitz_probability"] <= 1.0


def test_model_info_endpoint(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "was_blitz"
    assert "feature_order" in body
    assert "metrics" in body
    assert 0 <= body["metrics"]["roc_auc"] <= 1
    assert 0 <= body["metrics"]["accuracy"] <= 1


def test_predict_returns_valid_probability(client, sample_play):
    r = client.post("/predict", json=sample_play)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["blitz_probability"] <= 1.0
    assert isinstance(body["will_blitz"], bool)
    assert 0.0 < body["decision_threshold"] < 1.0
    assert body["cost_ratio_fn_to_fp"] > 0
    assert 0.0 < body["base_rate"] < 1.0
    assert body["recommendation"]


def test_predict_will_blitz_uses_tuned_threshold_not_half(client, sample_play):
    """will_blitz must follow the cost-tuned threshold from metadata, not 0.5."""
    r = client.post("/predict", json=sample_play)
    body = r.json()
    assert body["will_blitz"] == (body["blitz_probability"] >= body["decision_threshold"])


def test_response_drops_fabricated_fields(client, sample_play):
    """The model is binary and miscalibrated bands are meaningless -- neither a
    rusher-count estimate nor a confidence label should reappear."""
    body = client.post("/predict", json=sample_play).json()
    assert "num_pass_rushers_estimate" not in body
    assert "confidence" not in body


def test_lift_is_consistent_with_probability_and_base_rate(client, sample_play):
    body = client.post("/predict", json=sample_play).json()
    expected = body["blitz_probability"] / body["base_rate"]
    assert abs(body["lift_over_base_rate"] - expected) < 0.01


def test_probabilities_are_calibrated_not_raw(client, sample_play):
    """The raw model over-predicts badly (class_weight='balanced'). A calibrated
    response for an average-ish play should sit near the base rate, not near 0.5."""
    body = client.post("/predict", json=sample_play).json()
    assert body["blitz_probability"] < 0.45, (
        "probability looks uncalibrated -- is the calibrator being applied?")


def test_predict_rejects_missing_field(client, sample_play):
    bad = {k: v for k, v in sample_play.items() if k != "defenders_in_box"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # Pydantic validation error


def test_predict_rejects_out_of_range_down(client, sample_play):
    bad = {**sample_play, "down": 7}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_handles_unseen_team(client, sample_play):
    """Unknown team labels should encode to -1, not crash."""
    weird = {**sample_play, "possession_team": "ZZ", "defensive_team": "QQ"}
    r = client.post("/predict", json=weird)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["blitz_probability"] <= 1.0


def test_predict_batch(client, sample_play):
    second_play = {**sample_play, "down": 1, "yards_to_go": 10, "defenders_in_box": 5}
    r = client.post("/predict-batch", json={"plays": [sample_play, second_play]})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    for prediction in body:
        assert 0.0 <= prediction["blitz_probability"] <= 1.0


def test_predict_football_sanity_red_zone_heavy_box(client, sample_play):
    """Goal-line, heavy box, late game = high blitz probability."""
    goal_line = {
        **sample_play,
        "quarter": 4,
        "down": 4,
        "yards_to_go": 1,
        "yards_to_goal": 2,
        "defenders_in_box": 9,
        "game_clock": "1:45",
    }
    vanilla = {
        **sample_play,
        "quarter": 1,
        "down": 1,
        "yards_to_go": 10,
        "yards_to_goal": 75,
        "defenders_in_box": 5,
        "game_clock": "14:20",
    }
    heavy = client.post("/predict", json=goal_line).json()
    light = client.post("/predict", json=vanilla).json()

    # Assert the ordering and the lift, not an absolute probability. The previous
    # version of this test hard-coded p > 0.4, which was calibrated to the old
    # *uncalibrated* probabilities (inflated ~2x) and silently became far too strict
    # once real calibration landed. Relative behaviour is what "football-sensible"
    # actually means, and it survives recalibration and model swaps.
    assert heavy["blitz_probability"] > light["blitz_probability"], (
        f"9-man box at the goal line ({heavy['blitz_probability']}) should beat a "
        f"5-man box on an opening drive ({light['blitz_probability']})")
    assert heavy["lift_over_base_rate"] > 1.2, (
        f"Expected a clear lift over the base rate, got {heavy['lift_over_base_rate']}x")
    assert heavy["will_blitz"] is True, "goal-line heavy box should trip the alert"
    assert light["will_blitz"] is False, "vanilla opening drive should not trip the alert"
