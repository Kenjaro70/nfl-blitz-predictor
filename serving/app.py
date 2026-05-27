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
metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup, hold them until shutdown."""
    global model, label_encoders, metadata
    model = joblib.load(MODEL_DIR / "blitz_model.joblib")
    label_encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    yield


app = FastAPI(
    title="NFL Blitz Predictor",
    description="Predict whether the defense will blitz on a given pre-snap situation.",
    version="0.1.0",
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
    absolute_yardline_number: int = Field(..., ge=1, le=99, description="Yards from offense's goal line")
    defenders_in_box: int = Field(..., ge=0, le=11, description="Defenders within ~5 yards of LOS pre-snap")
    pre_snap_home_score: int = Field(..., ge=0)
    pre_snap_visitor_score: int = Field(..., ge=0)
    game_clock: str = Field(..., description="Time remaining in quarter, 'MM:SS'")
    is_home_offense: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    blitz_probability: float
    will_blitz: bool
    confidence: str
    num_pass_rushers_estimate: str


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
    is_red_zone = int(play.absolute_yardline_number <= 20)
    is_goal_to_go = int(play.yards_to_go >= play.absolute_yardline_number)

    row = {
        "possessionTeam": encode_categorical(play.possession_team, label_encoders["possessionTeam"]),
        "defensiveTeam": encode_categorical(play.defensive_team, label_encoders["defensiveTeam"]),
        "offenseFormation": encode_categorical(play.offense_formation, label_encoders["offenseFormation"]),
        "personnelO": encode_categorical(play.personnel_o, label_encoders["personnelO"]),
        "personnelD": encode_categorical(play.personnel_d, label_encoders["personnelD"]),
        "quarter": play.quarter,
        "down": play.down,
        "yardsToGo": play.yards_to_go,
        "absoluteYardlineNumber": play.absolute_yardline_number,
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


def describe_confidence(p: float) -> str:
    distance = abs(p - 0.5)
    if distance >= 0.35:
        return "very high"
    if distance >= 0.20:
        return "high"
    if distance >= 0.10:
        return "moderate"
    return "low"


def describe_rusher_estimate(p: float) -> str:
    if p < 0.25:
        return "likely 3-4 rushers (standard front)"
    if p < 0.5:
        return "likely 4 rushers, possible 5"
    if p < 0.75:
        return "likely 5 rushers (blitz)"
    return "likely 6+ rushers (heavy blitz)"


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
    proba = float(model.predict_proba(X)[0, 1])
    return PredictionResponse(
        blitz_probability=round(proba, 4),
        will_blitz=proba >= 0.5,
        confidence=describe_confidence(proba),
        num_pass_rushers_estimate=describe_rusher_estimate(proba),
    )


@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    rows = pd.concat([build_feature_row(p) for p in request.plays], ignore_index=True)
    probas = model.predict_proba(rows)[:, 1]
    return [
        PredictionResponse(
            blitz_probability=round(float(p), 4),
            will_blitz=bool(p >= 0.5),
            confidence=describe_confidence(p),
            num_pass_rushers_estimate=describe_rusher_estimate(p),
        )
        for p in probas
    ]
