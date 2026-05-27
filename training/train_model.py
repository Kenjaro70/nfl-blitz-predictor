"""Train the NFL pre-snap blitz classifier on Big Data Bowl 2023 data.

Expects the BDB 2023 CSVs in ../data/:
    games.csv, players.csv, plays.csv, pffScoutingData.csv,
    week1.csv ... week8.csv

The pff_role column in pffScoutingData.csv gives us per-player roles
(Pass Rush, Pass Coverage, Pass Block, etc.) which we use to count
pass rushers per play and define the blitz label.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Blitz = 5 or more pass rushers (standard NFL definition)
BLITZ_THRESHOLD = 5

CATEGORICAL_FEATURES = [
    "possessionTeam",
    "defensiveTeam",
    "offenseFormation",
    "personnelO",
    "personnelD",
]

NUMERIC_FEATURES = [
    "quarter",
    "down",
    "yardsToGo",
    "absoluteYardlineNumber",
    "defendersInBox",
    "preSnapHomeScore",
    "preSnapVisitorScore",
    "score_differential",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "is_home_offense",
    "is_two_minute",
    "is_red_zone",
    "is_goal_to_go",
]


def gameclock_to_seconds(clock: str) -> int:
    """'14:32' -> 872 (seconds remaining in the quarter)."""
    if pd.isna(clock):
        return 0
    parts = str(clock).split(":")
    if len(parts) != 2:
        return 0
    return int(parts[0]) * 60 + int(parts[1])


def build_blitz_labels(pff: pd.DataFrame) -> pd.DataFrame:
    """Count pass rushers per play and label blitz (>=5 rushers)."""
    rushers = (
        pff[pff["pff_role"] == "Pass Rush"]
        .groupby(["gameId", "playId"])
        .size()
        .rename("num_pass_rushers")
        .reset_index()
    )
    rushers["was_blitz"] = (rushers["num_pass_rushers"] >= BLITZ_THRESHOLD).astype(int)
    return rushers


def engineer_features(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Add derived pre-snap features."""
    df = plays.merge(games[["gameId", "homeTeamAbbr", "visitorTeamAbbr"]], on="gameId", how="left")

    df["is_home_offense"] = (df["possessionTeam"] == df["homeTeamAbbr"]).astype(int)

    df["score_differential"] = np.where(
        df["is_home_offense"] == 1,
        df["preSnapHomeScore"] - df["preSnapVisitorScore"],
        df["preSnapVisitorScore"] - df["preSnapHomeScore"],
    )

    df["clock_seconds"] = df["gameClock"].apply(gameclock_to_seconds)
    df["half_seconds_remaining"] = np.where(
        df["quarter"].isin([1, 3]),
        df["clock_seconds"] + 900,
        df["clock_seconds"],
    )
    df["game_seconds_remaining"] = np.where(
        df["quarter"] <= 2,
        df["half_seconds_remaining"] + 1800,
        df["half_seconds_remaining"],
    )

    df["is_two_minute"] = ((df["quarter"].isin([2, 4])) & (df["clock_seconds"] <= 120)).astype(int)
    df["is_red_zone"] = (df["absoluteYardlineNumber"] <= 20).astype(int)
    df["is_goal_to_go"] = (df["yardsToGo"] >= df["absoluteYardlineNumber"]).astype(int)

    return df


def main():
    print("Loading BDB 2023 data...")
    games = pd.read_csv(DATA_DIR / "games.csv")
    plays = pd.read_csv(DATA_DIR / "plays.csv")
    pff = pd.read_csv(DATA_DIR / "pffScoutingData.csv")

    print(f"  games:  {len(games):>7,} rows")
    print(f"  plays:  {len(plays):>7,} rows")
    print(f"  pff:    {len(pff):>7,} rows")

    print("\nBuilding blitz labels from pff_role...")
    labels = build_blitz_labels(pff)
    print(f"  labeled plays: {len(labels):>7,}")
    print(f"  blitz rate:    {labels['was_blitz'].mean():.1%}")

    print("\nMerging features and labels...")
    df = engineer_features(plays, games).merge(labels, on=["gameId", "playId"], how="inner")
    df = df.dropna(subset=CATEGORICAL_FEATURES + ["was_blitz"])
    print(f"  final dataset: {len(df):>7,} rows")

    print("\nEncoding categorical features...")
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df["was_blitz"]
    groups = df["gameId"]

    # Split by game so no game appears in both train and test.
    # Random play-level splits leak defensive scheme between halves.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"  train games: {groups.iloc[train_idx].nunique()}")
    print(f"  test games:  {groups.iloc[test_idx].nunique()}")
    print(f"  train plays: {len(X_train):,}  blitz rate: {y_train.mean():.1%}")
    print(f"  test plays:  {len(X_test):,}  blitz rate: {y_test.mean():.1%}")

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"  accuracy: {accuracy:.4f}")
    print(f"  ROC AUC:  {auc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["no_blitz", "blitz"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    importances = sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop 10 features by importance:")
    for name, imp in importances[:10]:
        print(f"  {name:<30} {imp:.4f}")

    print("\nSaving artifacts...")
    joblib.dump(model, MODEL_DIR / "blitz_model.joblib")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.joblib")

    metadata = {
        "model_type": "RandomForestClassifier",
        "target": "was_blitz",
        "blitz_definition": f">={BLITZ_THRESHOLD} pass rushers (pff_role == 'Pass Rush')",
        "feature_order": feature_cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "metrics": {
            "accuracy": float(accuracy),
            "roc_auc": float(auc),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "blitz_rate": float(y.mean()),
        },
        "data_source": "NFL Big Data Bowl 2023 (Kaggle)",
        "training_seasons": "2021 weeks 1-8",
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Artifacts in {MODEL_DIR}/")


if __name__ == "__main__":
    main()
