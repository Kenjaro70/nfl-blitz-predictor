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
        "absolute_yardline_number": 45,
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
    assert body["model"] == "RandomForestClassifier"


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
    assert body["confidence"] in {"very high", "high", "moderate", "low"}
    assert "rusher" in body["num_pass_rushers_estimate"].lower()


def test_predict_will_blitz_aligns_with_probability(client, sample_play):
    r = client.post("/predict", json=sample_play)
    body = r.json()
    assert body["will_blitz"] == (body["blitz_probability"] >= 0.5)


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
        "absolute_yardline_number": 2,
        "defenders_in_box": 9,
        "game_clock": "1:45",
    }
    r = client.post("/predict", json=goal_line)
    p = r.json()["blitz_probability"]
    # Not asserting a hard threshold the model might drift past in retraining,
    # but a 9-man box at the goal line should land above the overall blitz rate (~25%)
    assert p > 0.4, f"Expected high blitz prob in goal-line/heavy-box situation, got {p}"
