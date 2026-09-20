"""FastAPI serving layer for the NFL blitz predictor.

Loads the trained Random Forest, label encoders, and metadata at startup.
Accepts pre-snap offensive context and returns blitz probability.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).parent / "models"

model = None
label_encoders = None
calibrator = None
metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup, hold them until shutdown."""
    global model, label_encoders, calibrator, metadata
    model = joblib.load(MODEL_DIR / "blitz_model.joblib")
    label_encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
    calibrator = joblib.load(MODEL_DIR / "blitz_calibrator.joblib")
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    yield


app = FastAPI(
    title="NFL Blitz Predictor",
    description="Predict whether the defense will blitz on a given pre-snap situation.",
    # 0.3.0 keeps 0.2.0's request/response contract but swaps the model: logistic
    # regression instead of the random forest, since the two tie on AUC across by-game
    # splits. Same shape, different numbers -- the linear model is more conservative at
    # the extremes, so probabilities shift (usually down) for a given play.
    #
    # 0.2.0 was the breaking change from 0.1.0: absolute_yardline_number became
    # yards_to_goal, and the confidence / num_pass_rushers_estimate fields were removed.
    version="0.3.0",
    lifespan=lifespan,
)


class PlayContext(BaseModel):
    possession_team: str = Field(..., description="Offensive team abbr, e.g. 'KC'")
    defensive_team: str = Field(..., description="Defensive team abbr, e.g. 'BUF'")
    offense_formation: str = Field(..., description="e.g. 'SHOTGUN', 'SINGLEBACK', 'I_FORM'")
    personnel_o: str = Field(..., description="Offensive personnel, e.g. '1 RB, 1 TE, 3 WR'")
    personnel_d: str = Field(..., description="Defensive personnel, e.g. '4 DL, 2 LB, 5 DB'")
    quarter: int = Field(..., ge=1, le=5)
    down: int = Field(..., ge=1, le=4)
    yards_to_go: int = Field(..., ge=0, le=100)
    yards_to_goal: int = Field(
        ..., ge=1, le=99,
        description="Yards from the ball to the OPPONENT's goal line (1-99). "
                    "1 = about to score, 99 = backed up against own end zone. "
                    "Not the raw absoluteYardlineNumber from plays.csv, which is a "
                    "direction-dependent field coordinate; derive this from "
                    "yardlineSide + yardlineNumber instead.",
    )
    defenders_in_box: int = Field(..., ge=0, le=11, description="Defenders within ~5 yards of LOS pre-snap")
    pre_snap_home_score: int = Field(..., ge=0)
    pre_snap_visitor_score: int = Field(..., ge=0)
    game_clock: str = Field(..., description="Time remaining in quarter, 'MM:SS'")
    is_home_offense: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    blitz_probability: float = Field(
        ..., description="Calibrated probability the defense sends 5+ rushers. "
                         "Calibrated means 0.30 really does hit ~30% of the time.")
    will_blitz: bool = Field(
        ..., description="blitz_probability >= decision_threshold. This is an alert "
                         "flag tuned to a cost asymmetry, not a claim that a blitz is "
                         "more likely than not.")
    decision_threshold: float = Field(
        ..., description="Alert cutoff, from the cost ratio below.")
    cost_ratio_fn_to_fp: float = Field(
        ..., description="How much worse a missed blitz is than a false alarm. "
                         "Sets the threshold; change it and the threshold moves.")
    base_rate: float = Field(..., description="Blitz rate in the training data.")
    lift_over_base_rate: float = Field(
        ..., description="blitz_probability / base_rate. 1.0 means this situation is "
                         "no more blitz-prone than an average pass play.")
    recommendation: str


class BatchRequest(BaseModel):
    plays: List[PlayContext]


def gameclock_to_seconds(clock: str) -> int:
    parts = str(clock).split(":")
    if len(parts) != 2:
        return 0
    return int(parts[0]) * 60 + int(parts[1])


def encode_categorical(value: str, encoder) -> int:
    """Safely encode — return -1 for unseen labels rather than crashing."""
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return -1


def build_feature_row(play: PlayContext) -> pd.DataFrame:
    clock_seconds = gameclock_to_seconds(play.game_clock)
    half_seconds = clock_seconds + (900 if play.quarter in (1, 3) else 0)
    game_seconds = half_seconds + (1800 if play.quarter <= 2 else 0)

    score_diff = (
        play.pre_snap_home_score - play.pre_snap_visitor_score
        if play.is_home_offense
        else play.pre_snap_visitor_score - play.pre_snap_home_score
    )

    is_two_minute = int(play.quarter in (2, 4) and clock_seconds <= 120)
    is_red_zone = int(play.yards_to_goal <= 20)
    is_goal_to_go = int(play.yards_to_go >= play.yards_to_goal)

    row = {
        "possessionTeam": encode_categorical(play.possession_team, label_encoders["possessionTeam"]),
        "defensiveTeam": encode_categorical(play.defensive_team, label_encoders["defensiveTeam"]),
        "offenseFormation": encode_categorical(play.offense_formation, label_encoders["offenseFormation"]),
        "personnelO": encode_categorical(play.personnel_o, label_encoders["personnelO"]),
        "personnelD": encode_categorical(play.personnel_d, label_encoders["personnelD"]),
        "quarter": play.quarter,
        "down": play.down,
        "yardsToGo": play.yards_to_go,
        "yards_to_goal": play.yards_to_goal,
        "defendersInBox": play.defenders_in_box,
        "preSnapHomeScore": play.pre_snap_home_score,
        "preSnapVisitorScore": play.pre_snap_visitor_score,
        "score_differential": score_diff,
        "game_seconds_remaining": game_seconds,
        "half_seconds_remaining": half_seconds,
        "is_home_offense": play.is_home_offense,
        "is_two_minute": is_two_minute,
        "is_red_zone": is_red_zone,
        "is_goal_to_go": is_goal_to_go,
    }
    return pd.DataFrame([row])[metadata["feature_order"]]


def decision_rule() -> dict:
    return metadata.get("decision_rule", {})


def build_response(raw_proba: float) -> PredictionResponse:
    """Calibrate, apply the tuned threshold, and report both.

    Deliberately absent: a "confidence" label and an estimated rusher count. The
    model is binary and never predicts a count, and a confidence band derived from
    distance-to-0.5 says nothing about how trustworthy any single probability is.
    Both were removed rather than dressed up.
    """
    rule = decision_rule()
    threshold = float(rule.get("threshold", 0.5))
    cost_ratio = float(rule.get("cost_false_negative", 1.0)) / float(
        rule.get("cost_false_positive", 1.0))
    base_rate = float(metadata["metrics"]["blitz_rate"])

    proba = float(calibrator.predict([raw_proba])[0])
    alert = proba >= threshold

    if alert:
        recommendation = (
            f"Blitz probability {proba:.0%} is at or above the {threshold:.0%} alert "
            f"threshold (tuned for a {cost_ratio:.0f}:1 cost on missed blitzes). "
            "Favor extra protection."
        )
    else:
        recommendation = (
            f"Blitz probability {proba:.0%} is below the {threshold:.0%} alert "
            "threshold. Standard protection."
        )

    return PredictionResponse(
        blitz_probability=round(proba, 4),
        will_blitz=bool(alert),
        decision_threshold=round(threshold, 4),
        cost_ratio_fn_to_fp=cost_ratio,
        base_rate=round(base_rate, 4),
        lift_over_base_rate=round(proba / base_rate, 2),
        recommendation=recommendation,
    )


@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model": metadata.get("model_type") if metadata else None,
    }


@app.get("/model-info")
def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(play: PlayContext):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = build_feature_row(play)
    return build_response(float(model.predict_proba(X)[0, 1]))


@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    rows = pd.concat([build_feature_row(p) for p in request.plays], ignore_index=True)
    probas = model.predict_proba(rows)[:, 1]
    return [build_response(float(p)) for p in probas]
